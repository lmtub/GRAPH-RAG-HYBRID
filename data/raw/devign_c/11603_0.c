uint32_t HELPER(lcxbr)(CPUS390XState *env, uint32_t f1, uint32_t f2)

{

    CPU_QuadU x1, x2;



    x2.ll.upper = env->fregs[f2].ll;

    x2.ll.lower = env->fregs[f2 + 2].ll;

    x1.q = float128_chs(x2.q);

    env->fregs[f1].ll = x1.ll.upper;

    env->fregs[f1 + 2].ll = x1.ll.lower;

    return set_cc_nz_f128(x1.q);

}
