void helper_xssqrtqp(CPUPPCState *env, uint32_t opcode)

{

    ppc_vsr_t xb;

    ppc_vsr_t xt;

    float_status tstat;



    getVSR(rB(opcode) + 32, &xb, env);

    memset(&xt, 0, sizeof(xt));

    helper_reset_fpstatus(env);



    if (unlikely(Rc(opcode) != 0)) {

        /* TODO: Support xsadddpo after round-to-odd is implemented */

        abort();

    }



    tstat = env->fp_status;

    set_float_exception_flags(0, &tstat);

    xt.f128 = float128_sqrt(xb.f128, &tstat);

    env->fp_status.float_exception_flags |= tstat.float_exception_flags;



    if (unlikely(tstat.float_exception_flags & float_flag_invalid)) {

        if (float128_is_signaling_nan(xb.f128, &tstat)) {

            float_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN, 1);

            xt.f128 = float128_snan_to_qnan(xb.f128);

        } else if  (float128_is_quiet_nan(xb.f128, &tstat)) {

            xt.f128 = xb.f128;

        } else if (float128_is_neg(xb.f128) && !float128_is_zero(xb.f128)) {

            float_invalid_op_excp(env, POWERPC_EXCP_FP_VXSQRT, 1);

            set_snan_bit_is_one(0, &env->fp_status);

            xt.f128 = float128_default_nan(&env->fp_status);

        }

    }



    helper_compute_fprf_float128(env, xt.f128);

    putVSR(rD(opcode) + 32, &xt, env);

    float_check_status(env);

}
