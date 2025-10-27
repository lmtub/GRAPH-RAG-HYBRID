void cpu_io_recompile(CPUState *cpu, uintptr_t retaddr)

{

#if defined(TARGET_MIPS) || defined(TARGET_SH4)

    CPUArchState *env = cpu->env_ptr;

#endif

    TranslationBlock *tb;

    uint32_t n, cflags;

    target_ulong pc, cs_base;

    uint32_t flags;



    tb_lock();

    tb = tb_find_pc(retaddr);

    if (!tb) {

        cpu_abort(cpu, "cpu_io_recompile: could not find TB for pc=%p",

                  (void *)retaddr);

    }

    n = cpu->icount_decr.u16.low + tb->icount;

    cpu_restore_state_from_tb(cpu, tb, retaddr);

    /* Calculate how many instructions had been executed before the fault

       occurred.  */

    n = n - cpu->icount_decr.u16.low;

    /* Generate a new TB ending on the I/O insn.  */

    n++;

    /* On MIPS and SH, delay slot instructions can only be restarted if

       they were already the first instruction in the TB.  If this is not

       the first instruction in a TB then re-execute the preceding

       branch.  */

#if defined(TARGET_MIPS)

    if ((env->hflags & MIPS_HFLAG_BMASK) != 0 && n > 1) {

        env->active_tc.PC -= (env->hflags & MIPS_HFLAG_B16 ? 2 : 4);

        cpu->icount_decr.u16.low++;

        env->hflags &= ~MIPS_HFLAG_BMASK;

    }

#elif defined(TARGET_SH4)

    if ((env->flags & ((DELAY_SLOT | DELAY_SLOT_CONDITIONAL))) != 0

            && n > 1) {

        env->pc -= 2;

        cpu->icount_decr.u16.low++;

        env->flags &= ~(DELAY_SLOT | DELAY_SLOT_CONDITIONAL);

    }

#endif

    /* This should never happen.  */

    if (n > CF_COUNT_MASK) {

        cpu_abort(cpu, "TB too big during recompile");

    }



    cflags = n | CF_LAST_IO;

    cflags |= curr_cflags();

    pc = tb->pc;

    cs_base = tb->cs_base;

    flags = tb->flags;

    tb_phys_invalidate(tb, -1);

    if (tb->cflags & CF_NOCACHE) {

        if (tb->orig_tb) {

            /* Invalidate original TB if this TB was generated in

             * cpu_exec_nocache() */

            tb_phys_invalidate(tb->orig_tb, -1);

        }

        tb_free(tb);

    }

    /* FIXME: In theory this could raise an exception.  In practice

       we have already translated the block once so it's probably ok.  */

    tb_gen_code(cpu, pc, cs_base, flags, cflags);



    /* TODO: If env->pc != tb->pc (i.e. the faulting instruction was not

     * the first in the TB) then we end up generating a whole new TB and

     *  repeating the fault, which is horribly inefficient.

     *  Better would be to execute just this insn uncached, or generate a

     *  second new TB.

     *

     * cpu_loop_exit_noexc will longjmp back to cpu_exec where the

     * tb_lock gets reset.

     */

    cpu_loop_exit_noexc(cpu);

}
