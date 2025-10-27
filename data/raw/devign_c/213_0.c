static void cpu_exec_nocache(int max_cycles, TranslationBlock *orig_tb)

{

    unsigned long next_tb;

    TranslationBlock *tb;



    /* Should never happen.

       We only end up here when an existing TB is too long.  */

    if (max_cycles > CF_COUNT_MASK)

        max_cycles = CF_COUNT_MASK;



    tb = tb_gen_code(env, orig_tb->pc, orig_tb->cs_base, orig_tb->flags,

                     max_cycles);

    env->current_tb = tb;

    /* execute the generated code */

    next_tb = tcg_qemu_tb_exec(tb->tc_ptr);

    env->current_tb = NULL;



    if ((next_tb & 3) == 2) {

        /* Restore PC.  This may happen if async event occurs before

           the TB starts executing.  */

        cpu_pc_from_tb(env, tb);

    }

    tb_phys_invalidate(tb, -1);

    tb_free(tb);

}
