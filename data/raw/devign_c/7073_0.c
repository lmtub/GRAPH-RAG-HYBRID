float64 HELPER(ucf64_si2df)(float32 x, CPUUniCore32State *env)

{

    return int32_to_float64(ucf64_stoi(x), &env->ucf64.fp_status);

}
