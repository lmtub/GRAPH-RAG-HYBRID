int64_t helper_fdtox(CPUSPARCState *env, float64 src)

{

    int64_t ret;

    clear_float_exceptions(env);

    ret = float64_to_int64_round_to_zero(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
