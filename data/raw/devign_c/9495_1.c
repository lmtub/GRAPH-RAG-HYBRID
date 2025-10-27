static target_ulong remove_hpte(CPUPPCState *env, target_ulong ptex,

                                target_ulong avpn,

                                target_ulong flags,

                                target_ulong *vp, target_ulong *rp)

{

    uint8_t *hpte;

    target_ulong v, r, rb;



    if ((ptex * HASH_PTE_SIZE_64) & ~env->htab_mask) {

        return REMOVE_PARM;

    }



    hpte = env->external_htab + (ptex * HASH_PTE_SIZE_64);

    while (!lock_hpte(hpte, HPTE_V_HVLOCK)) {

        /* We have no real concurrency in qemu soft-emulation, so we

         * will never actually have a contested lock */

        assert(0);

    }



    v = ldq_p(hpte);

    r = ldq_p(hpte + (HASH_PTE_SIZE_64/2));



    if ((v & HPTE_V_VALID) == 0 ||

        ((flags & H_AVPN) && (v & ~0x7fULL) != avpn) ||

        ((flags & H_ANDCOND) && (v & avpn) != 0)) {

        stq_p(hpte, v & ~HPTE_V_HVLOCK);

        assert(!(ldq_p(hpte) & HPTE_V_HVLOCK));

        return REMOVE_NOT_FOUND;

    }

    *vp = v & ~HPTE_V_HVLOCK;

    *rp = r;

    stq_p(hpte, 0);

    rb = compute_tlbie_rb(v, r, ptex);

    ppc_tlb_invalidate_one(env, rb);

    assert(!(ldq_p(hpte) & HPTE_V_HVLOCK));

    return REMOVE_SUCCESS;

}
