void helper_store_fpcr (uint64_t val)

{

#ifdef CONFIG_SOFTFLOAT

    set_float_exception_flags((val >> 52) & 0x3F, &FP_STATUS);

#endif

    switch ((val >> 58) & 3) {

    case 0:

        set_float_rounding_mode(float_round_to_zero, &FP_STATUS);

        break;

    case 1:

        set_float_rounding_mode(float_round_down, &FP_STATUS);

        break;

    case 2:

        set_float_rounding_mode(float_round_nearest_even, &FP_STATUS);

        break;

    case 3:

        set_float_rounding_mode(float_round_up, &FP_STATUS);

        break;

    }

}
