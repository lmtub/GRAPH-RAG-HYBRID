uint32_t HELPER(ucf64_get_fpscr)(CPUUniCore32State *env)

{

    int i;

    uint32_t fpscr;



    fpscr = (env->ucf64.xregs[UC32_UCF64_FPSCR] & UCF64_FPSCR_MASK);

    i = get_float_exception_flags(&env->ucf64.fp_status);

    fpscr |= ucf64_exceptbits_from_host(i);

    return fpscr;

}
