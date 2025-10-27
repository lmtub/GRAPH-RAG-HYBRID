void cpu_tlb_update_dirty(CPUState *env)

{

    int i;

    for(i = 0; i < CPU_TLB_SIZE; i++)

        tlb_update_dirty(&env->tlb_table[0][i]);

    for(i = 0; i < CPU_TLB_SIZE; i++)

        tlb_update_dirty(&env->tlb_table[1][i]);

#if (NB_MMU_MODES >= 3)

    for(i = 0; i < CPU_TLB_SIZE; i++)

        tlb_update_dirty(&env->tlb_table[2][i]);

#endif

#if (NB_MMU_MODES >= 4)

    for(i = 0; i < CPU_TLB_SIZE; i++)

        tlb_update_dirty(&env->tlb_table[3][i]);

#endif

#if (NB_MMU_MODES >= 5)

    for(i = 0; i < CPU_TLB_SIZE; i++)

        tlb_update_dirty(&env->tlb_table[4][i]);

#endif

}
