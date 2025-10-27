static inline void cpu_loop_exec_tb(CPUState *cpu, TranslationBlock *tb,

                                    TranslationBlock **last_tb, int *tb_exit)

{

    uintptr_t ret;

    int32_t insns_left;



    if (unlikely(atomic_read(&cpu->exit_request))) {

        return;

    }



    trace_exec_tb(tb, tb->pc);

    ret = cpu_tb_exec(cpu, tb);

    tb = (TranslationBlock *)(ret & ~TB_EXIT_MASK);

    *tb_exit = ret & TB_EXIT_MASK;

    if (*tb_exit != TB_EXIT_REQUESTED) {

        *last_tb = tb;

        return;

    }



    *last_tb = NULL;

    insns_left = atomic_read(&cpu->icount_decr.u32);

    atomic_set(&cpu->icount_decr.u16.high, 0);

    if (insns_left < 0) {

        /* Something asked us to stop executing

         * chained TBs; just continue round the main

         * loop. Whatever requested the exit will also

         * have set something else (eg exit_request or

         * interrupt_request) which we will handle

         * next time around the loop.  But we need to

         * ensure the zeroing of tcg_exit_req (see cpu_tb_exec)

         * comes before the next read of cpu->exit_request

         * or cpu->interrupt_request.

         */

        smp_mb();

        return;

    }



    /* Instruction counter expired.  */

    assert(use_icount);

#ifndef CONFIG_USER_ONLY

    if (cpu->icount_extra) {

        /* Refill decrementer and continue execution.  */

        cpu->icount_extra += insns_left;

        insns_left = MIN(0xffff, cpu->icount_extra);

        cpu->icount_extra -= insns_left;

        cpu->icount_decr.u16.low = insns_left;

    } else {

        /* Execute any remaining instructions, then let the main loop

         * handle the next event.

         */

        if (insns_left > 0) {

            cpu_exec_nocache(cpu, insns_left, tb, false);

        }

    }

#endif

}
