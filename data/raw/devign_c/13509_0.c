uint32_t HELPER(lcdbr)(CPUS390XState *env, uint32_t f1, uint32_t f2)

{

    env->fregs[f1].d = float64_chs(env->fregs[f2].d);



    return set_cc_nz_f64(env->fregs[f1].d);

}
