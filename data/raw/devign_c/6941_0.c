void HELPER(ucf64_set_fpscr)(CPUUniCore32State *env, uint32_t val)

{

    int i;

    uint32_t changed;



    changed = env->ucf64.xregs[UC32_UCF64_FPSCR];

    env->ucf64.xregs[UC32_UCF64_FPSCR] = (val & UCF64_FPSCR_MASK);



    changed ^= val;

    if (changed & (UCF64_FPSCR_RND_MASK)) {

        i = UCF64_FPSCR_RND(val);

        switch (i) {

        case 0:

            i = float_round_nearest_even;

            break;

        case 1:

            i = float_round_to_zero;

            break;

        case 2:

            i = float_round_up;

            break;

        case 3:

            i = float_round_down;

            break;

        default: /* 100 and 101 not implement */

            cpu_abort(env, "Unsupported UniCore-F64 round mode");

        }

        set_float_rounding_mode(i, &env->ucf64.fp_status);

    }



    i = ucf64_exceptbits_to_host(UCF64_FPSCR_TRAPEN(val));

    set_float_exception_flags(i, &env->ucf64.fp_status);

}
