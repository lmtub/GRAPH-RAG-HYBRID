static RemoveResult remove_hpte(PowerPCCPU *cpu, target_ulong ptex,

                                target_ulong avpn,

                                target_ulong flags,

                                target_ulong *vp, target_ulong *rp)

{

    CPUPPCState *env = &cpu->env;

    uint64_t token;

    target_ulong v, r, rb;



    if (!valid_pte_index(env, ptex)) {

        return REMOVE_PARM;

    }



    token = ppc_hash64_start_access(cpu, ptex);

    v = ppc_hash64_load_hpte0(cpu, token, 0);

    r = ppc_hash64_load_hpte1(cpu, token, 0);

    ppc_hash64_stop_access(token);



    if ((v & HPTE64_V_VALID) == 0 ||

        ((flags & H_AVPN) && (v & ~0x7fULL) != avpn) ||

        ((flags & H_ANDCOND) && (v & avpn) != 0)) {

        return REMOVE_NOT_FOUND;

    }

    *vp = v;

    *rp = r;

    ppc_hash64_store_hpte(cpu, ptex, HPTE64_V_HPTE_DIRTY, 0);

    rb = compute_tlbie_rb(v, r, ptex);

    ppc_tlb_invalidate_one(env, rb);

    return REMOVE_SUCCESS;

}
