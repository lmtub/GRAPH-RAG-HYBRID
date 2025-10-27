static target_ulong h_protect(CPUPPCState *env, sPAPREnvironment *spapr,

                              target_ulong opcode, target_ulong *args)

{

    target_ulong flags = args[0];

    target_ulong pte_index = args[1];

    target_ulong avpn = args[2];

    uint8_t *hpte;

    target_ulong v, r, rb;



    if ((pte_index * HASH_PTE_SIZE_64) & ~env->htab_mask) {

        return H_PARAMETER;

    }



    hpte = env->external_htab + (pte_index * HASH_PTE_SIZE_64);

    while (!lock_hpte(hpte, HPTE_V_HVLOCK)) {

        /* We have no real concurrency in qemu soft-emulation, so we

         * will never actually have a contested lock */

        assert(0);

    }



    v = ldq_p(hpte);

    r = ldq_p(hpte + (HASH_PTE_SIZE_64/2));



    if ((v & HPTE_V_VALID) == 0 ||

        ((flags & H_AVPN) && (v & ~0x7fULL) != avpn)) {

        stq_p(hpte, v & ~HPTE_V_HVLOCK);

        assert(!(ldq_p(hpte) & HPTE_V_HVLOCK));

        return H_NOT_FOUND;

    }



    r &= ~(HPTE_R_PP0 | HPTE_R_PP | HPTE_R_N |

           HPTE_R_KEY_HI | HPTE_R_KEY_LO);

    r |= (flags << 55) & HPTE_R_PP0;

    r |= (flags << 48) & HPTE_R_KEY_HI;

    r |= flags & (HPTE_R_PP | HPTE_R_N | HPTE_R_KEY_LO);

    rb = compute_tlbie_rb(v, r, pte_index);

    stq_p(hpte, v & ~HPTE_V_VALID);

    ppc_tlb_invalidate_one(env, rb);

    stq_p(hpte + (HASH_PTE_SIZE_64/2), r);

    /* Don't need a memory barrier, due to qemu's global lock */

    stq_p(hpte, v & ~HPTE_V_HVLOCK);

    assert(!(ldq_p(hpte) & HPTE_V_HVLOCK));

    return H_SUCCESS;

}
