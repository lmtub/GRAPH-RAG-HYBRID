uint64_t helper_frsp(CPUPPCState *env, uint64_t arg)

{

    CPU_DoubleU farg;

    float32 f32;



    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN square root */

        fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN);

    }

    f32 = float64_to_float32(farg.d, &env->fp_status);

    farg.d = float32_to_float64(f32, &env->fp_status);



    return farg.ll;

}
