void qmp_inject_nmi(Error **errp)

{

#if defined(TARGET_I386)

    CPUState *cs;



    CPU_FOREACH(cs) {

        X86CPU *cpu = X86_CPU(cs);



        if (!cpu->apic_state) {

            cpu_interrupt(cs, CPU_INTERRUPT_NMI);

        } else {

            apic_deliver_nmi(cpu->apic_state);

        }

    }

#elif defined(TARGET_S390X)

    CPUState *cs;

    S390CPU *cpu;



    CPU_FOREACH(cs) {

        cpu = S390_CPU(cs);

        if (cpu->env.cpu_num == monitor_get_cpu_index()) {

            if (s390_cpu_restart(S390_CPU(cs)) == -1) {

                error_set(errp, QERR_UNSUPPORTED);

                return;

            }

            break;

        }

    }

#else

    error_set(errp, QERR_UNSUPPORTED);

#endif

}
