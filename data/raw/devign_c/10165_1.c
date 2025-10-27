static int handle_intercept(S390CPU *cpu)

{

    CPUState *cs = CPU(cpu);

    struct kvm_run *run = cs->kvm_run;

    int icpt_code = run->s390_sieic.icptcode;

    int r = 0;



    DPRINTF("intercept: 0x%x (at 0x%lx)\n", icpt_code,

            (long)cs->kvm_run->psw_addr);

    switch (icpt_code) {

        case ICPT_INSTRUCTION:

            r = handle_instruction(cpu, run);

            break;

        case ICPT_WAITPSW:

            /* disabled wait, since enabled wait is handled in kernel */

            if (s390_del_running_cpu(cpu) == 0) {

                if (is_special_wait_psw(cs)) {

                    qemu_system_shutdown_request();

                } else {

                    QObject *data;



                    data = qobject_from_jsonf("{ 'action': %s }", "pause");

                    monitor_protocol_event(QEVENT_GUEST_PANICKED, data);

                    qobject_decref(data);

                    vm_stop(RUN_STATE_GUEST_PANICKED);

                }

            }

            r = EXCP_HALTED;

            break;

        case ICPT_CPU_STOP:

            if (s390_del_running_cpu(cpu) == 0) {

                qemu_system_shutdown_request();

            }

            r = EXCP_HALTED;

            break;

        case ICPT_SOFT_INTERCEPT:

            fprintf(stderr, "KVM unimplemented icpt SOFT\n");

            exit(1);

            break;

        case ICPT_IO:

            fprintf(stderr, "KVM unimplemented icpt IO\n");

            exit(1);

            break;

        default:

            fprintf(stderr, "Unknown intercept code: %d\n", icpt_code);

            exit(1);

            break;

    }



    return r;

}
