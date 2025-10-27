float64 HELPER(ucf64_muld)(float64 a, float64 b, CPUUniCore32State *env)

{

    return float64_mul(a, b, &env->ucf64.fp_status);

}
