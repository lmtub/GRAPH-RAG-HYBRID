int cpu_arm_handle_mmu_fault (CPUState *env, target_ulong address,

                              int access_type, int mmu_idx, int is_softmmu)

{

    uint32_t phys_addr;

    target_ulong page_size;

    int prot;

    int ret, is_user;



    is_user = mmu_idx == MMU_USER_IDX;

    ret = get_phys_addr(env, address, access_type, is_user, &phys_addr, &prot,

                        &page_size);

    if (ret == 0) {

        /* Map a single [sub]page.  */

        phys_addr &= ~(uint32_t)0x3ff;

        address &= ~(uint32_t)0x3ff;

        tlb_set_page (env, address, phys_addr, prot | PAGE_EXEC, mmu_idx,

                      page_size);

        return 0;

    }



    if (access_type == 2) {

        env->cp15.c5_insn = ret;

        env->cp15.c6_insn = address;

        env->exception_index = EXCP_PREFETCH_ABORT;

    } else {

        env->cp15.c5_data = ret;

        if (access_type == 1 && arm_feature(env, ARM_FEATURE_V6))

            env->cp15.c5_data |= (1 << 11);

        env->cp15.c6_data = address;

        env->exception_index = EXCP_DATA_ABORT;

    }

    return 1;

}
