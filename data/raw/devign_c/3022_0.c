float64 helper_fitod(CPUSPARCState *env, int32_t src)

{

    /* No possible exceptions converting int to double.  */

    return int32_to_float64(src, &env->fp_status);

}
