float32 helper_fxtos(CPUSPARCState *env, int64_t src)

{

    float32 ret;

    clear_float_exceptions(env);

    ret = int64_to_float32(src, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
