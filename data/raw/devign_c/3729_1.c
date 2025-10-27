uint64_t helper_mulldo(CPUPPCState *env, uint64_t arg1, uint64_t arg2)

{

    int64_t th;

    uint64_t tl;



    muls64(&tl, (uint64_t *)&th, arg1, arg2);

    /* If th != 0 && th != -1, then we had an overflow */

    if (likely((uint64_t)(th + 1) <= 1)) {

        env->ov = 0;

    } else {

        env->so = env->ov = 1;

    }

    return (int64_t)tl;

}
