target_ulong helper_rdhwr_cpunum(CPUMIPSState *env)

{

    check_hwrena(env, 0);

    return env->CP0_EBase & 0x3ff;

}
