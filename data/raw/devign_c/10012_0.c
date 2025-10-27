target_ulong helper_rdhwr_cpunum(CPUMIPSState *env)

{

    if ((env->hflags & MIPS_HFLAG_CP0) ||

        (env->CP0_HWREna & (1 << 0)))

        return env->CP0_EBase & 0x3ff;

    else

        do_raise_exception(env, EXCP_RI, GETPC());



    return 0;

}
