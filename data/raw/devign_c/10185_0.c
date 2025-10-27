static void cpu_exec_nocache(CPUState *cpu, int max_cycles,

                             TranslationBlock *orig_tb)

{

    TranslationBlock *tb;



    /* Should never happen.

       We only end up here when an existing TB is too long.  */

    if (max_cycles > CF_COUNT_MASK)

        max_cycles = CF_COUNT_MASK;



    tb = tb_gen_code(cpu, orig_tb->pc, orig_tb->cs_base, orig_tb->flags,

                     max_cycles | CF_NOCACHE);

    tb->orig_tb = tcg_ctx.tb_ctx.tb_invalidated_flag ? NULL : orig_tb;

    cpu->current_tb = tb;

    /* execute the generated code */

    trace_exec_tb_nocache(tb, tb->pc);

    cpu_tb_exec(cpu, tb->tc_ptr);

    cpu->current_tb = NULL;

    tb_phys_invalidate(tb, -1);

    tb_free(tb);

}
