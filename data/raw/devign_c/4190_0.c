static inline int ucf64_exceptbits_from_host(int host_bits)

{

    int target_bits = 0;



    if (host_bits & float_flag_invalid) {

        target_bits |= UCF64_FPSCR_FLAG_INVALID;

    }

    if (host_bits & float_flag_divbyzero) {

        target_bits |= UCF64_FPSCR_FLAG_DIVZERO;

    }

    if (host_bits & float_flag_overflow) {

        target_bits |= UCF64_FPSCR_FLAG_OVERFLOW;

    }

    if (host_bits & float_flag_underflow) {

        target_bits |= UCF64_FPSCR_FLAG_UNDERFLOW;

    }

    if (host_bits & float_flag_inexact) {

        target_bits |= UCF64_FPSCR_FLAG_INEXACT;

    }

    return target_bits;

}
