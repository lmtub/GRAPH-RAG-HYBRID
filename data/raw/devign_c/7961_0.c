static target_ulong h_enter(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                            target_ulong opcode, target_ulong *args)

{

    CPUPPCState *env = &cpu->env;

    target_ulong flags = args[0];

    target_ulong pte_index = args[1];

    target_ulong pteh = args[2];

    target_ulong ptel = args[3];

    unsigned apshift, spshift;

    target_ulong raddr;

    target_ulong index;

    uint64_t token;



    apshift = ppc_hash64_hpte_page_shift_noslb(cpu, pteh, ptel, &spshift);

    if (!apshift) {

        /* Bad page size encoding */

        return H_PARAMETER;

    }



    raddr = (ptel & HPTE64_R_RPN) & ~((1ULL << apshift) - 1);



    if (is_ram_address(spapr, raddr)) {

        /* Regular RAM - should have WIMG=0010 */

        if ((ptel & HPTE64_R_WIMG) != HPTE64_R_M) {

            return H_PARAMETER;

        }

    } else {

        /* Looks like an IO address */

        /* FIXME: What WIMG combinations could be sensible for IO?

         * For now we allow WIMG=010x, but are there others? */

        /* FIXME: Should we check against registered IO addresses? */

        if ((ptel & (HPTE64_R_W | HPTE64_R_I | HPTE64_R_M)) != HPTE64_R_I) {

            return H_PARAMETER;

        }

    }



    pteh &= ~0x60ULL;



    if (!valid_pte_index(env, pte_index)) {

        return H_PARAMETER;

    }



    index = 0;

    if (likely((flags & H_EXACT) == 0)) {

        pte_index &= ~7ULL;

        token = ppc_hash64_start_access(cpu, pte_index);

        for (; index < 8; index++) {

            if (!(ppc_hash64_load_hpte0(cpu, token, index) & HPTE64_V_VALID)) {

                break;

            }

        }

        ppc_hash64_stop_access(token);

        if (index == 8) {

            return H_PTEG_FULL;

        }

    } else {

        token = ppc_hash64_start_access(cpu, pte_index);

        if (ppc_hash64_load_hpte0(cpu, token, 0) & HPTE64_V_VALID) {

            ppc_hash64_stop_access(token);

            return H_PTEG_FULL;

        }

        ppc_hash64_stop_access(token);

    }



    ppc_hash64_store_hpte(cpu, pte_index + index,

                          pteh | HPTE64_V_HPTE_DIRTY, ptel);



    args[0] = pte_index + index;

    return H_SUCCESS;

}
