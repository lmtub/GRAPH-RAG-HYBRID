static inline void cpu_loop_exec_tb(CPUState *cpu, TranslationBlock *tb,

                                    TranslationBlock **last_tb, int *tb_exit,

                                    SyncClocks *sc)

{

    uintptr_t ret;



    if (unlikely(atomic_read(&cpu->exit_request))) {

        return;

    }



    trace_exec_tb(tb, tb->pc);

    ret = cpu_tb_exec(cpu, tb);

    *last_tb = (TranslationBlock *)(ret & ~TB_EXIT_MASK);

    *tb_exit = ret & TB_EXIT_MASK;

    switch (*tb_exit) {

    case TB_EXIT_REQUESTED:

        /* Something asked us to stop executing

         * chained TBs; just continue round the main

         * loop. Whatever requested the exit will also

         * have set something else (eg exit_request or

         * interrupt_request) which we will handle

         * next time around the loop.  But we need to

         * ensure the tcg_exit_req read in generated code

         * comes before the next read of cpu->exit_request

         * or cpu->interrupt_request.

         */

        smp_rmb();

        *last_tb = NULL;

        break;

    case TB_EXIT_ICOUNT_EXPIRED:

    {

        /* Instruction counter expired.  */

#ifdef CONFIG_USER_ONLY

        abort();

#else

        int insns_left = cpu->icount_decr.u32;

        if (cpu->icount_extra && insns_left >= 0) {

            /* Refill decrementer and continue execution.  */

            cpu->icount_extra += insns_left;

            insns_left = MIN(0xffff, cpu->icount_extra);

            cpu->icount_extra -= insns_left;

            cpu->icount_decr.u16.low = insns_left;

        } else {

            if (insns_left > 0) {

                /* Execute remaining instructions.  */

                cpu_exec_nocache(cpu, insns_left, *last_tb, false);

                align_clocks(sc, cpu);

            }

            cpu->exception_index = EXCP_INTERRUPT;

            *last_tb = NULL;

            cpu_loop_exit(cpu);

        }

        break;

#endif

    }

    default:

        break;

    }

}
