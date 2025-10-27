float64 helper_fxtod(CPUSPARCState *env, int64_t src)

{

    float64 ret;

    clear_float_exceptions(env);

    ret = int64_to_float64(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
