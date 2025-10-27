static always_inline void fload_invalid_op_excp (int op)

{

    int ve;



    ve = fpscr_ve;

    if (op & POWERPC_EXCP_FP_VXSNAN) {

        /* Operation on signaling NaN */

        env->fpscr |= 1 << FPSCR_VXSNAN;

    }

    if (op & POWERPC_EXCP_FP_VXSOFT) {

        /* Software-defined condition */

        env->fpscr |= 1 << FPSCR_VXSOFT;

    }

    switch (op & ~(POWERPC_EXCP_FP_VXSOFT | POWERPC_EXCP_FP_VXSNAN)) {

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

            FT0 = (uint64_t)-1;

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

            FT0 = (uint64_t)-1;

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

        if (msr_fe0 != 0 || msr_fe1 != 0)

            do_raise_exception_err(POWERPC_EXCP_PROGRAM, POWERPC_EXCP_FP | op);

    }

}
