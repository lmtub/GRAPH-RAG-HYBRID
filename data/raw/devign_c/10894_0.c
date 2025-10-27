void s390_cpu_do_interrupt(CPUState *cs)

{

    S390CPU *cpu = S390_CPU(cs);

    CPUS390XState *env = &cpu->env;



    qemu_log_mask(CPU_LOG_INT, "%s: %d at pc=%" PRIx64 "\n",

                  __func__, cs->exception_index, env->psw.addr);



    s390_cpu_set_state(CPU_STATE_OPERATING, cpu);

    /* handle machine checks */

    if ((env->psw.mask & PSW_MASK_MCHECK) &&

        (cs->exception_index == -1)) {

        if (env->pending_int & INTERRUPT_MCHK) {

            cs->exception_index = EXCP_MCHK;

        }

    }

    /* handle external interrupts */

    if ((env->psw.mask & PSW_MASK_EXT) &&

        cs->exception_index == -1 &&

        (env->pending_int & INTERRUPT_EXT)) {

        cs->exception_index = EXCP_EXT;

    }

    /* handle I/O interrupts */

    if ((env->psw.mask & PSW_MASK_IO) &&

        (cs->exception_index == -1)) {

        if (env->pending_int & INTERRUPT_IO) {

            cs->exception_index = EXCP_IO;

        }

    }



    switch (cs->exception_index) {

    case EXCP_PGM:

        do_program_interrupt(env);

        break;

    case EXCP_SVC:

        do_svc_interrupt(env);

        break;

    case EXCP_EXT:

        do_ext_interrupt(env);

        break;

    case EXCP_IO:

        do_io_interrupt(env);

        break;

    case EXCP_MCHK:

        do_mchk_interrupt(env);

        break;

    }

    cs->exception_index = -1;



    if (!env->pending_int) {

        cs->interrupt_request &= ~CPU_INTERRUPT_HARD;

    }

}
