target_ulong helper_rdhwr_ccres(CPUMIPSState *env)

{

    check_hwrena(env, 3);

    return env->CCRes;

}
