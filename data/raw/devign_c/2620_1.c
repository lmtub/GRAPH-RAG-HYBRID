void tlb_reset_dirty(CPUState *cpu, ram_addr_t start1, ram_addr_t length)

{

    CPUArchState *env;



    int mmu_idx;



    assert_cpu_is_self(cpu);



    env = cpu->env_ptr;

    for (mmu_idx = 0; mmu_idx < NB_MMU_MODES; mmu_idx++) {

        unsigned int i;



        for (i = 0; i < CPU_TLB_SIZE; i++) {

            tlb_reset_dirty_range(&env->tlb_table[mmu_idx][i],

                                  start1, length);

        }



        for (i = 0; i < CPU_VTLB_SIZE; i++) {

            tlb_reset_dirty_range(&env->tlb_v_table[mmu_idx][i],

                                  start1, length);

        }

    }

}
