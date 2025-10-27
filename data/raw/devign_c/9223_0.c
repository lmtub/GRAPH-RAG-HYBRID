target_ulong helper_yield(target_ulong arg1)

{

    if (arg1 < 0) {

        /* No scheduling policy implemented. */

        if (arg1 != -2) {

            if (env->CP0_VPEControl & (1 << CP0VPECo_YSI) &&

                env->active_tc.CP0_TCStatus & (1 << CP0TCSt_DT)) {

                env->CP0_VPEControl &= ~(0x7 << CP0VPECo_EXCPT);

                env->CP0_VPEControl |= 4 << CP0VPECo_EXCPT;

                helper_raise_exception(EXCP_THREAD);

            }

        }

    } else if (arg1 == 0) {

        if (0 /* TODO: TC underflow */) {

            env->CP0_VPEControl &= ~(0x7 << CP0VPECo_EXCPT);

            helper_raise_exception(EXCP_THREAD);

        } else {

            // TODO: Deallocate TC

        }

    } else if (arg1 > 0) {

        /* Yield qualifier inputs not implemented. */

        env->CP0_VPEControl &= ~(0x7 << CP0VPECo_EXCPT);

        env->CP0_VPEControl |= 2 << CP0VPECo_EXCPT;

        helper_raise_exception(EXCP_THREAD);

    }

    return env->CP0_YQMask;

}
