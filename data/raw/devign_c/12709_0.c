static bool linked_bp_matches(ARMCPU *cpu, int lbn)

{

    CPUARMState *env = &cpu->env;

    uint64_t bcr = env->cp15.dbgbcr[lbn];

    int brps = extract32(cpu->dbgdidr, 24, 4);

    int ctx_cmps = extract32(cpu->dbgdidr, 20, 4);

    int bt;

    uint32_t contextidr;



    /* Links to unimplemented or non-context aware breakpoints are

     * CONSTRAINED UNPREDICTABLE: either behave as if disabled, or

     * as if linked to an UNKNOWN context-aware breakpoint (in which

     * case DBGWCR<n>_EL1.LBN must indicate that breakpoint).

     * We choose the former.

     */

    if (lbn > brps || lbn < (brps - ctx_cmps)) {

        return false;

    }



    bcr = env->cp15.dbgbcr[lbn];



    if (extract64(bcr, 0, 1) == 0) {

        /* Linked breakpoint disabled : generate no events */

        return false;

    }



    bt = extract64(bcr, 20, 4);



    /* We match the whole register even if this is AArch32 using the

     * short descriptor format (in which case it holds both PROCID and ASID),

     * since we don't implement the optional v7 context ID masking.

     */

    contextidr = extract64(env->cp15.contextidr_el1, 0, 32);



    switch (bt) {

    case 3: /* linked context ID match */

        if (arm_current_el(env) > 1) {

            /* Context matches never fire in EL2 or (AArch64) EL3 */

            return false;

        }

        return (contextidr == extract64(env->cp15.dbgbvr[lbn], 0, 32));

    case 5: /* linked address mismatch (reserved in AArch64) */

    case 9: /* linked VMID match (reserved if no EL2) */

    case 11: /* linked context ID and VMID match (reserved if no EL2) */

    default:

        /* Links to Unlinked context breakpoints must generate no

         * events; we choose to do the same for reserved values too.

         */

        return false;

    }



    return false;

}
