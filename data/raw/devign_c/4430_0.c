uint32_t helper_efdctuf (uint64_t val)

{

    CPU_DoubleU u;

    float64 tmp;



    u.ll = val;

    /* NaN are not treated the same way IEEE 754 does */

    if (unlikely(float64_is_nan(u.d)))

        return 0;

    tmp = uint64_to_float64(1ULL << 32, &env->vec_status);

    u.d = float64_mul(u.d, tmp, &env->vec_status);



    return float64_to_uint32(u.d, &env->vec_status);

}
