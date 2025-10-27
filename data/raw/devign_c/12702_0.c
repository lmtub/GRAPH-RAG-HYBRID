uint64_t helper_cmptun (uint64_t a, uint64_t b)

{

    float64 fa, fb;



    fa = t_to_float64(a);

    fb = t_to_float64(b);



    if (float64_is_quiet_nan(fa) || float64_is_quiet_nan(fb))

        return 0x4000000000000000ULL;

    else

        return 0;

}
