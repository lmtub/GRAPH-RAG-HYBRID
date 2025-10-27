int mips_cpu_handle_mmu_fault(CPUState *cs, vaddr address, int rw,

                              int mmu_idx)

{

    MIPSCPU *cpu = MIPS_CPU(cs);

    CPUMIPSState *env = &cpu->env;

#if !defined(CONFIG_USER_ONLY)

    hwaddr physical;

    int prot;

    int access_type;

#endif

    int ret = 0;



#if 0

    log_cpu_state(cs, 0);

#endif

    qemu_log_mask(CPU_LOG_MMU,

              "%s pc " TARGET_FMT_lx " ad %" VADDR_PRIx " rw %d mmu_idx %d\n",

              __func__, env->active_tc.PC, address, rw, mmu_idx);



    /* data access */

#if !defined(CONFIG_USER_ONLY)

    /* XXX: put correct access by using cpu_restore_state()

       correctly */

    access_type = ACCESS_INT;

    ret = get_physical_address(env, &physical, &prot,

                               address, rw, access_type);

    qemu_log_mask(CPU_LOG_MMU,

             "%s address=%" VADDR_PRIx " ret %d physical " TARGET_FMT_plx

             " prot %d\n",

             __func__, address, ret, physical, prot);

    if (ret == TLBRET_MATCH) {

        tlb_set_page(cs, address & TARGET_PAGE_MASK,

                     physical & TARGET_PAGE_MASK, prot | PAGE_EXEC,

                     mmu_idx, TARGET_PAGE_SIZE);

        ret = 0;

    } else if (ret < 0)

#endif

    {

        raise_mmu_exception(env, address, rw, ret);

        ret = 1;

    }



    return ret;

}
