static uint32_t cas_check_pvr(PowerPCCPU *cpu, target_ulong *addr,

                              Error **errp)

{

    bool explicit_match = false; /* Matched the CPU's real PVR */

    uint32_t max_compat = cpu->max_compat;

    uint32_t best_compat = 0;

    int i;



    /*

     * We scan the supplied table of PVRs looking for two things

     *   1. Is our real CPU PVR in the list?

     *   2. What's the "best" listed logical PVR

     */

    for (i = 0; i < 512; ++i) {

        uint32_t pvr, pvr_mask;



        pvr_mask = ldl_be_phys(&address_space_memory, *addr);

        pvr = ldl_be_phys(&address_space_memory, *addr + 4);

        *addr += 8;



        if (~pvr_mask & pvr) {

            break; /* Terminator record */

        }



        if ((cpu->env.spr[SPR_PVR] & pvr_mask) == (pvr & pvr_mask)) {

            explicit_match = true;

        } else {

            if (ppc_check_compat(cpu, pvr, best_compat, max_compat)) {

                best_compat = pvr;

            }

        }

    }



    if ((best_compat == 0) && (!explicit_match || max_compat)) {

        /* We couldn't find a suitable compatibility mode, and either

         * the guest doesn't support "raw" mode for this CPU, or raw

         * mode is disabled because a maximum compat mode is set */

        error_setg(errp, "Couldn't negotiate a suitable PVR during CAS");

        return 0;

    }



    /* Parsing finished */

    trace_spapr_cas_pvr(cpu->compat_pvr, explicit_match, best_compat);



    return best_compat;

}
