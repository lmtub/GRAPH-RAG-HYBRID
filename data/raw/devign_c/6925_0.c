void HELPER(access_check_cp_reg)(CPUARMState *env, void *rip, uint32_t syndrome)

{

    const ARMCPRegInfo *ri = rip;

    int target_el;



    if (arm_feature(env, ARM_FEATURE_XSCALE) && ri->cp < 14

        && extract32(env->cp15.c15_cpar, ri->cp, 1) == 0) {

        raise_exception(env, EXCP_UDEF, syndrome, exception_target_el(env));

    }



    if (!ri->accessfn) {

        return;

    }



    switch (ri->accessfn(env, ri)) {

    case CP_ACCESS_OK:

        return;

    case CP_ACCESS_TRAP:

        target_el = exception_target_el(env);

        break;

    case CP_ACCESS_TRAP_EL2:

        /* Requesting a trap to EL2 when we're in EL3 or S-EL0/1 is

         * a bug in the access function.

         */

        assert(!arm_is_secure(env) && !arm_current_el(env) == 3);

        target_el = 2;

        break;

    case CP_ACCESS_TRAP_EL3:

        target_el = 3;

        break;

    case CP_ACCESS_TRAP_UNCATEGORIZED:

        target_el = exception_target_el(env);

        syndrome = syn_uncategorized();

        break;

    default:

        g_assert_not_reached();

    }



    raise_exception(env, EXCP_UDEF, syndrome, target_el);

}
