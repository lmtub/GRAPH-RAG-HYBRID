uint64_t helper_fsqrt(CPUPPCState *env, uint64_t arg)

{

    CPU_DoubleU farg;



    farg.ll = arg;



    if (unlikely(float64_is_neg(farg.d) && !float64_is_zero(farg.d))) {

        /* Square root of a negative nonzero number */

        farg.ll = fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSQRT);

    } else {

        if (unlikely(float64_is_signaling_nan(farg.d))) {

            /* sNaN square root */

            fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN);

        }

        farg.d = float64_sqrt(farg.d, &env->fp_status);

    }

    return farg.ll;

}
