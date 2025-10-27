void helper_single_step(CPUX86State *env)

{

#ifndef CONFIG_USER_ONLY

    check_hw_breakpoints(env, 1);

    env->dr[6] |= DR6_BS;

#endif

    raise_exception(env, EXCP01_DB);

}
