float32 HELPER(ucf64_adds)(float32 a, float32 b, CPUUniCore32State *env)

{

    return float32_add(a, b, &env->ucf64.fp_status);

}
