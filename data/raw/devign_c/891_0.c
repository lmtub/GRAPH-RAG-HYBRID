uint32_t helper_efdctuiz (uint64_t val)

{

    CPU_DoubleU u;



    u.ll = val;

    /* NaN are not treated the same way IEEE 754 does */

    if (unlikely(float64_is_nan(u.d)))

        return 0;



    return float64_to_uint32_round_to_zero(u.d, &env->vec_status);

}
