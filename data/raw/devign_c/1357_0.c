float32 HELPER(ucf64_muls)(float32 a, float32 b, CPUUniCore32State *env)

{

    return float32_mul(a, b, &env->ucf64.fp_status);

}
