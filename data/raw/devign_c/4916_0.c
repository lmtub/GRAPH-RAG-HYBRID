static void float64_maddsub_update_excp(CPUPPCState *env, float64 arg1,

                                        float64 arg2, float64 arg3,

                                        unsigned int madd_flags)

{

    if (unlikely((float64_is_infinity(arg1) && float64_is_zero(arg2)) ||

                 (float64_is_zero(arg1) && float64_is_infinity(arg2)))) {

        /* Multiplication of zero by infinity */

        arg1 = float_invalid_op_excp(env, POWERPC_EXCP_FP_VXIMZ, 1);

    } else if (unlikely(float64_is_signaling_nan(arg1, &env->fp_status) ||

                        float64_is_signaling_nan(arg2, &env->fp_status) ||

                        float64_is_signaling_nan(arg3, &env->fp_status))) {

        /* sNaN operation */

        float_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN, 1);

    } else if ((float64_is_infinity(arg1) || float64_is_infinity(arg2)) &&

               float64_is_infinity(arg3)) {

        uint8_t aSign, bSign, cSign;



        aSign = float64_is_neg(arg1);

        bSign = float64_is_neg(arg2);

        cSign = float64_is_neg(arg3);

        if (madd_flags & float_muladd_negate_c) {

            cSign ^= 1;

        }

        if (aSign ^ bSign ^ cSign) {

            float_invalid_op_excp(env, POWERPC_EXCP_FP_VXISI, 1);

        }

    }

}
