static void fpu_init (CPUMIPSState *env, const mips_def_t *def)

{

    int i;



    for (i = 0; i < MIPS_FPU_MAX; i++)

        env->fpus[i].fcr0 = def->CP1_fcr0;



    memcpy(&env->active_fpu, &env->fpus[0], sizeof(env->active_fpu));

    if (env->user_mode_only) {

        if (env->CP0_Config1 & (1 << CP0C1_FP))

            env->hflags |= MIPS_HFLAG_FPU;

#ifdef TARGET_MIPS64

        if (env->active_fpu.fcr0 & (1 << FCR0_F64))

            env->hflags |= MIPS_HFLAG_F64;

#endif

    }

}
