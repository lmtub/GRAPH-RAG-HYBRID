static inline int ucf64_exceptbits_to_host(int target_bits)

{

    int host_bits = 0;



    if (target_bits & UCF64_FPSCR_FLAG_INVALID) {

        host_bits |= float_flag_invalid;

    }

    if (target_bits & UCF64_FPSCR_FLAG_DIVZERO) {

        host_bits |= float_flag_divbyzero;

    }

    if (target_bits & UCF64_FPSCR_FLAG_OVERFLOW) {

        host_bits |= float_flag_overflow;

    }

    if (target_bits & UCF64_FPSCR_FLAG_UNDERFLOW) {

        host_bits |= float_flag_underflow;

    }

    if (target_bits & UCF64_FPSCR_FLAG_INEXACT) {

        host_bits |= float_flag_inexact;

    }

    return host_bits;

}
