float64 helper_fqtod(CPUSPARCState *env)

{

    float64 ret;

    clear_float_exceptions(env);

    ret = float128_to_float64(QT1, &env->fp_status);

    check_ieee_exceptions(env);

    return ret;

}
