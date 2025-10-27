static inline bool cpu_handle_exception(CPUState *cpu, int *ret)

{

    if (cpu->exception_index >= 0) {

        if (cpu->exception_index >= EXCP_INTERRUPT) {

            /* exit request from the cpu execution loop */

            *ret = cpu->exception_index;

            if (*ret == EXCP_DEBUG) {

                cpu_handle_debug_exception(cpu);

            }

            cpu->exception_index = -1;

            return true;

        } else {

#if defined(CONFIG_USER_ONLY)

            /* if user mode only, we simulate a fake exception

               which will be handled outside the cpu execution

               loop */

#if defined(TARGET_I386)

            CPUClass *cc = CPU_GET_CLASS(cpu);

            cc->do_interrupt(cpu);

#endif

            *ret = cpu->exception_index;

            cpu->exception_index = -1;

            return true;

#else

            if (replay_exception()) {

                CPUClass *cc = CPU_GET_CLASS(cpu);

                qemu_mutex_lock_iothread();

                cc->do_interrupt(cpu);

                qemu_mutex_unlock_iothread();

                cpu->exception_index = -1;

            } else if (!replay_has_interrupt()) {

                /* give a chance to iothread in replay mode */

                *ret = EXCP_INTERRUPT;

                return true;

            }

#endif

        }

#ifndef CONFIG_USER_ONLY

    } else if (replay_has_exception()

               && cpu->icount_decr.u16.low + cpu->icount_extra == 0) {

        /* try to cause an exception pending in the log */

        cpu_exec_nocache(cpu, 1, tb_find(cpu, NULL, 0), true);

        *ret = -1;

        return true;

#endif

    }



    return false;

}
