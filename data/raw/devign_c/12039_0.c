int kvm_cpu_exec(CPUState *env)

{

    struct kvm_run *run = env->kvm_run;

    int ret;



    DPRINTF("kvm_cpu_exec()\n");



    if (kvm_arch_process_irqchip_events(env)) {

        env->exit_request = 0;

        env->exception_index = EXCP_HLT;

        return 0;

    }



    do {

        if (env->kvm_vcpu_dirty) {

            kvm_arch_put_registers(env, KVM_PUT_RUNTIME_STATE);

            env->kvm_vcpu_dirty = 0;

        }



        kvm_arch_pre_run(env, run);

        if (env->exit_request) {

            DPRINTF("interrupt exit requested\n");

            /*

             * KVM requires us to reenter the kernel after IO exits to complete

             * instruction emulation. This self-signal will ensure that we

             * leave ASAP again.

             */

            qemu_cpu_kick_self();

        }

        cpu_single_env = NULL;

        qemu_mutex_unlock_iothread();



        ret = kvm_vcpu_ioctl(env, KVM_RUN, 0);



        qemu_mutex_lock_iothread();

        cpu_single_env = env;

        kvm_arch_post_run(env, run);



        kvm_flush_coalesced_mmio_buffer();



        if (ret == -EINTR || ret == -EAGAIN) {

            cpu_exit(env);

            DPRINTF("io window exit\n");

            ret = 0;

            break;

        }



        if (ret < 0) {

            DPRINTF("kvm run failed %s\n", strerror(-ret));

            abort();

        }



        ret = 0; /* exit loop */

        switch (run->exit_reason) {

        case KVM_EXIT_IO:

            DPRINTF("handle_io\n");

            kvm_handle_io(run->io.port,

                          (uint8_t *)run + run->io.data_offset,

                          run->io.direction,

                          run->io.size,

                          run->io.count);

            ret = 1;

            break;

        case KVM_EXIT_MMIO:

            DPRINTF("handle_mmio\n");

            cpu_physical_memory_rw(run->mmio.phys_addr,

                                   run->mmio.data,

                                   run->mmio.len,

                                   run->mmio.is_write);

            ret = 1;

            break;

        case KVM_EXIT_IRQ_WINDOW_OPEN:

            DPRINTF("irq_window_open\n");

            break;

        case KVM_EXIT_SHUTDOWN:

            DPRINTF("shutdown\n");

            qemu_system_reset_request();

            ret = 1;

            break;

        case KVM_EXIT_UNKNOWN:

            fprintf(stderr, "KVM: unknown exit, hardware reason %" PRIx64 "\n",

                    (uint64_t)run->hw.hardware_exit_reason);

            ret = -1;

            break;

#ifdef KVM_CAP_INTERNAL_ERROR_DATA

        case KVM_EXIT_INTERNAL_ERROR:

            ret = kvm_handle_internal_error(env, run);

            break;

#endif

        case KVM_EXIT_DEBUG:

            DPRINTF("kvm_exit_debug\n");

#ifdef KVM_CAP_SET_GUEST_DEBUG

            if (kvm_arch_debug(&run->debug.arch)) {

                env->exception_index = EXCP_DEBUG;

                return 0;

            }

            /* re-enter, this exception was guest-internal */

            ret = 1;

#endif /* KVM_CAP_SET_GUEST_DEBUG */

            break;

        default:

            DPRINTF("kvm_arch_handle_exit\n");

            ret = kvm_arch_handle_exit(env, run);

            break;

        }

    } while (ret > 0);



    if (ret < 0) {

        cpu_dump_state(env, stderr, fprintf, CPU_DUMP_CODE);

        vm_stop(0);

        env->exit_request = 1;

    }

    if (env->exit_request) {

        env->exit_request = 0;

        env->exception_index = EXCP_INTERRUPT;

    }



    return ret;

}
