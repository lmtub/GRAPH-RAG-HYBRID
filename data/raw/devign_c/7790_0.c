int cpu_mips_handle_mmu_fault (CPUState *env, target_ulong address, int rw,

                               int mmu_idx, int is_softmmu)

{

#if !defined(CONFIG_USER_ONLY)

    target_phys_addr_t physical;

    int prot;

#endif

    int access_type;

    int ret = 0;



#if 0

    log_cpu_state(env, 0);

#endif

    qemu_log("%s pc " TARGET_FMT_lx " ad " TARGET_FMT_lx " rw %d mmu_idx %d smmu %d\n",

              __func__, env->active_tc.PC, address, rw, mmu_idx, is_softmmu);



    rw &= 1;



    /* data access */

    /* XXX: put correct access by using cpu_restore_state()

       correctly */

    access_type = ACCESS_INT;

#if defined(CONFIG_USER_ONLY)

    ret = TLBRET_NOMATCH;

#else

    ret = get_physical_address(env, &physical, &prot,

                               address, rw, access_type);

    qemu_log("%s address=" TARGET_FMT_lx " ret %d physical " TARGET_FMT_plx " prot %d\n",

              __func__, address, ret, physical, prot);

    if (ret == TLBRET_MATCH) {

       ret = tlb_set_page(env, address & TARGET_PAGE_MASK,

                          physical & TARGET_PAGE_MASK, prot,

                          mmu_idx, is_softmmu);

    } else if (ret < 0)

#endif

    {

        raise_mmu_exception(env, address, rw, ret);

        ret = 1;

    }



    return ret;

}
