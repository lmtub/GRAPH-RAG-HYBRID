void r4k_helper_tlbwi(CPUMIPSState *env)

{

    r4k_tlb_t *tlb;

    int idx;

    target_ulong VPN;

    uint16_t ASID;

    bool G, V0, D0, V1, D1;



    idx = (env->CP0_Index & ~0x80000000) % env->tlb->nb_tlb;

    tlb = &env->tlb->mmu.r4k.tlb[idx];

    VPN = env->CP0_EntryHi & (TARGET_PAGE_MASK << 1);

#if defined(TARGET_MIPS64)

    VPN &= env->SEGMask;

#endif

    ASID = env->CP0_EntryHi & env->CP0_EntryHi_ASID_mask;

    G = env->CP0_EntryLo0 & env->CP0_EntryLo1 & 1;

    V0 = (env->CP0_EntryLo0 & 2) != 0;

    D0 = (env->CP0_EntryLo0 & 4) != 0;

    V1 = (env->CP0_EntryLo1 & 2) != 0;

    D1 = (env->CP0_EntryLo1 & 4) != 0;



    /* Discard cached TLB entries, unless tlbwi is just upgrading access

       permissions on the current entry. */

    if (tlb->VPN != VPN || tlb->ASID != ASID || tlb->G != G ||

        (tlb->V0 && !V0) || (tlb->D0 && !D0) ||

        (tlb->V1 && !V1) || (tlb->D1 && !D1)) {

        r4k_mips_tlb_flush_extra(env, env->tlb->nb_tlb);

    }



    r4k_invalidate_tlb(env, idx, 0);

    r4k_fill_tlb(env, idx);

}
