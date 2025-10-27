void tlb_flush(CPUState *env, int flush_global)

{

    int i;



#if defined(DEBUG_TLB)

    printf("tlb_flush:\n");

#endif

    /* must reset current TB so that interrupts cannot modify the

       links while we are modifying them */

    env->current_tb = NULL;



    for(i = 0; i < CPU_TLB_SIZE; i++) {

        env->tlb_table[0][i].addr_read = -1;

        env->tlb_table[0][i].addr_write = -1;

        env->tlb_table[0][i].addr_code = -1;

        env->tlb_table[1][i].addr_read = -1;

        env->tlb_table[1][i].addr_write = -1;

        env->tlb_table[1][i].addr_code = -1;

#if (NB_MMU_MODES >= 3)

        env->tlb_table[2][i].addr_read = -1;

        env->tlb_table[2][i].addr_write = -1;

        env->tlb_table[2][i].addr_code = -1;

#if (NB_MMU_MODES == 4)

        env->tlb_table[3][i].addr_read = -1;

        env->tlb_table[3][i].addr_write = -1;

        env->tlb_table[3][i].addr_code = -1;

#endif

#endif

    }



    memset (env->tb_jmp_cache, 0, TB_JMP_CACHE_SIZE * sizeof (void *));



#ifdef USE_KQEMU

    if (env->kqemu_enabled) {

        kqemu_flush(env, flush_global);

    }

#endif

    tlb_flush_count++;

}
