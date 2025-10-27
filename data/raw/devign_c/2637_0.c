static inline uint64_t fload_invalid_op_excp(CPUPPCState *env, int op)

{

    uint64_t ret = 0;

    int ve;



    ve = fpscr_ve;

    switch (op) {

    case POWERPC_EXCP_FP_VXSNAN:

        env->fpscr |= 1 << FPSCR_VXSNAN;

        break;

    case POWERPC_EXCP_FP_VXSOFT:

        env->fpscr |= 1 << FPSCR_VXSOFT;

        break;

    case POWERPC_EXCP_FP_VXISI:

        /* Magnitude subtraction of infinities */

        env->fpscr |= 1 << FPSCR_VXISI;

        goto update_arith;

    case POWERPC_EXCP_FP_VXIDI:

        /* Division of infinity by infinity */

        env->fpscr |= 1 << FPSCR_VXIDI;

        goto update_arith;

    case POWERPC_EXCP_FP_VXZDZ:

        /* Division of zero by zero */

        env->fpscr |= 1 << FPSCR_VXZDZ;

        goto update_arith;

    case POWERPC_EXCP_FP_VXIMZ:

        /* Multiplication of zero by infinity */

        env->fpscr |= 1 << FPSCR_VXIMZ;

        goto update_arith;

    case POWERPC_EXCP_FP_VXVC:

        /* Ordered comparison of NaN */

        env->fpscr |= 1 << FPSCR_VXVC;

        env->fpscr &= ~(0xF << FPSCR_FPCC);

        env->fpscr |= 0x11 << FPSCR_FPCC;

        /* We must update the target FPR before raising the exception */

        if (ve != 0) {

            env->exception_index = POWERPC_EXCP_PROGRAM;

            env->error_code = POWERPC_EXCP_FP | POWERPC_EXCP_FP_VXVC;

            /* Update the floating-point enabled exception summary */

            env->fpscr |= 1 << FPSCR_FEX;

            /* Exception is differed */

            ve = 0;

        }

        break;

    case POWERPC_EXCP_FP_VXSQRT:

        /* Square root of a negative number */

        env->fpscr |= 1 << FPSCR_VXSQRT;

    update_arith:

        env->fpscr &= ~((1 << FPSCR_FR) | (1 << FPSCR_FI));

        if (ve == 0) {

            /* Set the result to quiet NaN */

            ret = 0x7FF8000000000000ULL;

            env->fpscr &= ~(0xF << FPSCR_FPCC);

            env->fpscr |= 0x11 << FPSCR_FPCC;

        }

        break;

    case POWERPC_EXCP_FP_VXCVI:

        /* Invalid conversion */

        env->fpscr |= 1 << FPSCR_VXCVI;

        env->fpscr &= ~((1 << FPSCR_FR) | (1 << FPSCR_FI));

        if (ve == 0) {

            /* Set the result to quiet NaN */

            ret = 0x7FF8000000000000ULL;

            env->fpscr &= ~(0xF << FPSCR_FPCC);

            env->fpscr |= 0x11 << FPSCR_FPCC;

        }

        break;

    }

    /* Update the floating-point invalid operation summary */

    env->fpscr |= 1 << FPSCR_VX;

    /* Update the floating-point exception summary */

    env->fpscr |= 1 << FPSCR_FX;

    if (ve != 0) {

        /* Update the floating-point enabled exception summary */

        env->fpscr |= 1 << FPSCR_FEX;

        if (msr_fe0 != 0 || msr_fe1 != 0) {

            helper_raise_exception_err(env, POWERPC_EXCP_PROGRAM,

                                       POWERPC_EXCP_FP | op);

        }

    }

    return ret;

}
