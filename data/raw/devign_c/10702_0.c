float64 HELPER(ucf64_subd)(float64 a, float64 b, CPUUniCore32State *env)

{

    return float64_sub(a, b, &env->ucf64.fp_status);

}
