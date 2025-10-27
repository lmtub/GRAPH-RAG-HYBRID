static inline int check_ap(CPUARMState *env, ARMMMUIdx mmu_idx,

                           int ap, int domain_prot,

                           int access_type)

{

    int prot_ro;

    bool is_user = regime_is_user(env, mmu_idx);



    if (domain_prot == 3) {

        return PAGE_READ | PAGE_WRITE;

    }



    if (access_type == 1) {

        prot_ro = 0;

    } else {

        prot_ro = PAGE_READ;

    }



    switch (ap) {

    case 0:

        if (arm_feature(env, ARM_FEATURE_V7)) {

            return 0;

        }

        if (access_type == 1) {

            return 0;

        }

        switch (regime_sctlr(env, mmu_idx) & (SCTLR_S | SCTLR_R)) {

        case SCTLR_S:

            return is_user ? 0 : PAGE_READ;

        case SCTLR_R:

            return PAGE_READ;

        default:

            return 0;

        }

    case 1:

        return is_user ? 0 : PAGE_READ | PAGE_WRITE;

    case 2:

        if (is_user) {

            return prot_ro;

        } else {

            return PAGE_READ | PAGE_WRITE;

        }

    case 3:

        return PAGE_READ | PAGE_WRITE;

    case 4: /* Reserved.  */

        return 0;

    case 5:

        return is_user ? 0 : prot_ro;

    case 6:

        return prot_ro;

    case 7:

        if (!arm_feature(env, ARM_FEATURE_V6K)) {

            return 0;

        }

        return prot_ro;

    default:

        abort();

    }

}
