void helper_fxtoq(CPUSPARCState *env, int64_t src)

{

    /* No possible exceptions converting long long to long double.  */

    QT0 = int64_to_float128(src, &env->fp_status);

}
