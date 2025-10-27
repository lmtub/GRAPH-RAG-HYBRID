int kvm_cpu_exec(CPUState *cpu)

{

    struct kvm_run *run = cpu->kvm_run;

    int ret, run_ret;



    DPRINTF("kvm_cpu_exec()\n");



    if (kvm_arch_process_async_events(cpu)) {

        cpu->exit_request = 0;

        return EXCP_HLT;

    }



    qemu_mutex_unlock_iothread();



    do {

        MemTxAttrs attrs;



        if (cpu->kvm_vcpu_dirty) {

            kvm_arch_put_registers(cpu, KVM_PUT_RUNTIME_STATE);

            cpu->kvm_vcpu_dirty = false;

        }



        kvm_arch_pre_run(cpu, run);

        if (cpu->exit_request) {

            DPRINTF("interrupt exit requested\n");

            /*

             * KVM requires us to reenter the kernel after IO exits to complete

             * instruction emulation. This self-signal will ensure that we

             * leave ASAP again.

             */

            qemu_cpu_kick_self();

        }



        run_ret = kvm_vcpu_ioctl(cpu, KVM_RUN, 0);



        attrs = kvm_arch_post_run(cpu, run);



        if (run_ret < 0) {

            if (run_ret == -EINTR || run_ret == -EAGAIN) {

                DPRINTF("io window exit\n");

                ret = EXCP_INTERRUPT;

                break;

            }

            fprintf(stderr, "error: kvm run failed %s\n",

                    strerror(-run_ret));

#ifdef TARGET_PPC

            if (run_ret == -EBUSY) {

                fprintf(stderr,

                        "This is probably because your SMT is enabled.\n"

                        "VCPU can only run on primary threads with all "

                        "secondary threads offline.\n");

            }

#endif

            ret = -1;

            break;

        }



        trace_kvm_run_exit(cpu->cpu_index, run->exit_reason);

        switch (run->exit_reason) {

        case KVM_EXIT_IO:

            DPRINTF("handle_io\n");

            /* Called outside BQL */

            kvm_handle_io(run->io.port, attrs,

                          (uint8_t *)run + run->io.data_offset,

                          run->io.direction,

                          run->io.size,

                          run->io.count);

            ret = 0;

            break;

        case KVM_EXIT_MMIO:

            DPRINTF("handle_mmio\n");

            /* Called outside BQL */

            address_space_rw(&address_space_memory,

                             run->mmio.phys_addr, attrs,

                             run->mmio.data,

                             run->mmio.len,

                             run->mmio.is_write);

            ret = 0;

            break;

        case KVM_EXIT_IRQ_WINDOW_OPEN:

            DPRINTF("irq_window_open\n");

            ret = EXCP_INTERRUPT;

            break;

        case KVM_EXIT_SHUTDOWN:

            DPRINTF("shutdown\n");

            qemu_system_reset_request();

            ret = EXCP_INTERRUPT;

            break;

        case KVM_EXIT_UNKNOWN:

            fprintf(stderr, "KVM: unknown exit, hardware reason %" PRIx64 "\n",

                    (uint64_t)run->hw.hardware_exit_reason);

            ret = -1;

            break;

        case KVM_EXIT_INTERNAL_ERROR:

            ret = kvm_handle_internal_error(cpu, run);

            break;

        case KVM_EXIT_SYSTEM_EVENT:

            switch (run->system_event.type) {

            case KVM_SYSTEM_EVENT_SHUTDOWN:

                qemu_system_shutdown_request();

                ret = EXCP_INTERRUPT;

                break;

            case KVM_SYSTEM_EVENT_RESET:

                qemu_system_reset_request();

                ret = EXCP_INTERRUPT;

                break;

            case KVM_SYSTEM_EVENT_CRASH:


                qemu_mutex_lock_iothread();

                qemu_system_guest_panicked();

                qemu_mutex_unlock_iothread();

                ret = 0;

                break;

            default:

                DPRINTF("kvm_arch_handle_exit\n");

                ret = kvm_arch_handle_exit(cpu, run);

                break;

            }

            break;

        default:

            DPRINTF("kvm_arch_handle_exit\n");

            ret = kvm_arch_handle_exit(cpu, run);

            break;

        }

    } while (ret == 0);



    qemu_mutex_lock_iothread();



    if (ret < 0) {

        cpu_dump_state(cpu, stderr, fprintf, CPU_DUMP_CODE);

        vm_stop(RUN_STATE_INTERNAL_ERROR);

    }



    cpu->exit_request = 0;

    return ret;

}