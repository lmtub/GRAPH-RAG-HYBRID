hwaddr ppc_hash64_get_phys_page_debug(PowerPCCPU *cpu, target_ulong addr)

{

    CPUPPCState *env = &cpu->env;

    ppc_slb_t *slb;

    hwaddr pte_offset;

    ppc_hash_pte64_t pte;

    unsigned apshift;



    if (msr_dr == 0) {

        /* In real mode the top 4 effective address bits are ignored */

        return addr & 0x0FFFFFFFFFFFFFFFULL;

    }



    slb = slb_lookup(cpu, addr);

    if (!slb) {

        return -1;

    }



    pte_offset = ppc_hash64_htab_lookup(cpu, slb, addr, &pte);

    if (pte_offset == -1) {

        return -1;

    }



    apshift = hpte_page_shift(slb->sps, pte.pte0, pte.pte1);

    if (!apshift) {

        return -1;

    }



    return deposit64(pte.pte1 & HPTE64_R_RPN, 0, apshift, addr)

        & TARGET_PAGE_MASK;

}
