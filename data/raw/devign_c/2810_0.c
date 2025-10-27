static inline uint64_t do_fri(uint64_t arg, int rounding_mode)

{

    CPU_DoubleU farg;

    farg.ll = arg;



    if (unlikely(float64_is_signaling_nan(farg.d))) {

        /* sNaN round */

        farg.ll = fload_invalid_op_excp(POWERPC_EXCP_FP_VXSNAN | POWERPC_EXCP_FP_VXCVI);

    } else if (unlikely(float64_is_nan(farg.d) || float64_is_infinity(farg.d))) {

        /* qNan / infinity round */

        farg.ll = fload_invalid_op_excp(POWERPC_EXCP_FP_VXCVI);

    } else {

        set_float_rounding_mode(rounding_mode, &env->fp_status);

        farg.ll = float64_round_to_int(farg.d, &env->fp_status);

        /* Restore rounding mode from FPSCR */

        fpscr_set_rounding_mode();

    }

    return farg.ll;

}
