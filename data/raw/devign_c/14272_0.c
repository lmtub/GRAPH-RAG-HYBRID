uint64_t helper_fre(CPUPPCState *env, uint64_t arg)

{

    CPU_DoubleU farg;



    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN reciprocal */

        fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN);

    }

    farg.d = float64_div(float64_one, farg.d, &env->fp_status);

    return farg.d;

}
