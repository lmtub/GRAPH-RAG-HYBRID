static RemoveResult remove_hpte(CPUPPCState *env, target_ulong ptex,

                                target_ulong avpn,

                                target_ulong flags,

                                target_ulong *vp, target_ulong *rp)

{

    hwaddr hpte;

    target_ulong v, r, rb;



    if ((ptex * HASH_PTE_SIZE_64) & ~env->htab_mask) {

        return REMOVE_PARM;

    }



    hpte = ptex * HASH_PTE_SIZE_64;



    v = ppc_hash64_load_hpte0(env, hpte);

    r = ppc_hash64_load_hpte1(env, hpte);



    if ((v & HPTE64_V_VALID) == 0 ||

        ((flags & H_AVPN) && (v & ~0x7fULL) != avpn) ||

        ((flags & H_ANDCOND) && (v & avpn) != 0)) {

        return REMOVE_NOT_FOUND;

    }

    *vp = v;

    *rp = r;

    ppc_hash64_store_hpte0(env, hpte, HPTE64_V_HPTE_DIRTY);

    rb = compute_tlbie_rb(v, r, ptex);

    ppc_tlb_invalidate_one(env, rb);

    return REMOVE_SUCCESS;

}
