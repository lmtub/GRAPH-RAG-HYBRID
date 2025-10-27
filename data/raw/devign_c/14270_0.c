void helper_fcmpo(CPUPPCState *env, uint64_t arg1, uint64_t arg2,

                  uint32_t crfD)

{

    CPU_DoubleU farg1, farg2;

    uint32_t ret = 0;



    farg1.ll = arg1;

    farg2.ll = arg2;



    if (unlikely(float64_is_any_nan(farg1.d) ||

                 float64_is_any_nan(farg2.d))) {

        ret = 0x01UL;

    } else if (float64_lt(farg1.d, farg2.d, &env->fp_status)) {

        ret = 0x08UL;

    } else if (!float64_le(farg1.d, farg2.d, &env->fp_status)) {

        ret = 0x04UL;

    } else {

        ret = 0x02UL;

    }



    env->fpscr &= ~(0x0F << FPSCR_FPRF);

    env->fpscr |= ret << FPSCR_FPRF;

    env->crf[crfD] = ret;

    if (unlikely(ret == 0x01UL)) {

        if (float64_is_signaling_nan(farg1.d) ||

            float64_is_signaling_nan(farg2.d)) {

            /* sNaN comparison */

            fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXSNAN |

                                  POWERPC_EXCP_FP_VXVC);

        } else {

            /* qNaN comparison */

            fload_invalid_op_excp(env, POWERPC_EXCP_FP_VXVC);

        }

    }

}
