void HELPER(mvpg)(CPUS390XState *env, uint64_t r0, uint64_t r1, uint64_t r2)

{

    /* XXX missing r0 handling */

    env->cc_op = 0;

#ifdef CONFIG_USER_ONLY

    memmove(g2h(r1), g2h(r2), TARGET_PAGE_SIZE);

#else

    mvc_fast_memmove(env, TARGET_PAGE_SIZE, r1, r2);

#endif

}
