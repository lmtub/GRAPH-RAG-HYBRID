int s390_cpu_handle_mmu_fault(CPUState *cs, vaddr orig_vaddr,

                              int rw, int mmu_idx)

{

    S390CPU *cpu = S390_CPU(cs);

    CPUS390XState *env = &cpu->env;

    uint64_t asc = cpu_mmu_idx_to_asc(mmu_idx);

    target_ulong vaddr, raddr;

    int prot;



    DPRINTF("%s: address 0x%" VADDR_PRIx " rw %d mmu_idx %d\n",

            __func__, orig_vaddr, rw, mmu_idx);



    orig_vaddr &= TARGET_PAGE_MASK;

    vaddr = orig_vaddr;



    /* 31-Bit mode */

    if (!(env->psw.mask & PSW_MASK_64)) {

        vaddr &= 0x7fffffff;

    }



    if (mmu_translate(env, vaddr, rw, asc, &raddr, &prot, true)) {

        /* Translation ended in exception */

        return 1;

    }



    /* check out of RAM access */

    if (raddr > ram_size) {

        DPRINTF("%s: raddr %" PRIx64 " > ram_size %" PRIx64 "\n", __func__,

                (uint64_t)raddr, (uint64_t)ram_size);

        trigger_pgm_exception(env, PGM_ADDRESSING, ILEN_AUTO);

        return 1;

    }



    qemu_log_mask(CPU_LOG_MMU, "%s: set tlb %" PRIx64 " -> %" PRIx64 " (%x)\n",

            __func__, (uint64_t)vaddr, (uint64_t)raddr, prot);



    tlb_set_page(cs, orig_vaddr, raddr, prot,

                 mmu_idx, TARGET_PAGE_SIZE);



    return 0;

}
