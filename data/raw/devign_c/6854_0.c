uint64_t helper_fres (uint64_t arg)

{

    CPU_DoubleU fone, farg;

    float32 f32;

    fone.ll = 0x3FF0000000000000ULL; /* 1.0 */

    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN reciprocal */

        farg.ll = fload_invalid_op_excp(POWERPC_EXCP_FP_VXSNAN);

    } else if (unlikely(float64_is_zero(farg.d))) {

        /* Zero reciprocal */

        farg.ll = float_zero_divide_excp(fone.d, farg.d);

    } else {

        farg.d = float64_div(fone.d, farg.d, &env->fp_status);

        f32 = float64_to_float32(farg.d, &env->fp_status);

        farg.d = float32_to_float64(f32, &env->fp_status);

    }

    return farg.ll;

}
