void cpu_reset (CPUMIPSState *env)

{

    memset(env, 0, offsetof(CPUMIPSState, breakpoints));



    tlb_flush(env, 1);



    /* Minimal init */

#if !defined(CONFIG_USER_ONLY)

    if (env->hflags & MIPS_HFLAG_BMASK) {

        /* If the exception was raised from a delay slot,

         * come back to the jump.  */

        env->CP0_ErrorEPC = env->PC - 4;

        env->hflags &= ~MIPS_HFLAG_BMASK;

    } else {

        env->CP0_ErrorEPC = env->PC;

    }

    env->PC = (int32_t)0xBFC00000;

#if defined (MIPS_USES_R4K_TLB)

    env->CP0_Random = MIPS_TLB_NB - 1;

    env->tlb_in_use = MIPS_TLB_NB;

#endif

    env->CP0_Wired = 0;

    /* SMP not implemented */

    env->CP0_EBase = (int32_t)0x80000000;

    env->CP0_Config0 = MIPS_CONFIG0;

    env->CP0_Config1 = MIPS_CONFIG1;

    env->CP0_Config2 = MIPS_CONFIG2;

    env->CP0_Config3 = MIPS_CONFIG3;

    env->CP0_Status = (1 << CP0St_BEV) | (1 << CP0St_ERL);

    env->CP0_WatchLo = 0;

    env->hflags = MIPS_HFLAG_ERL;

    /* Count register increments in debug mode, EJTAG version 1 */

    env->CP0_Debug = (1 << CP0DB_CNT) | (0x1 << CP0DB_VER);

    env->CP0_PRid = MIPS_CPU;

#endif

    env->exception_index = EXCP_NONE;

#if defined(CONFIG_USER_ONLY)

    env->hflags |= MIPS_HFLAG_UM;

    env->user_mode_only = 1;

#endif

#ifdef MIPS_USES_FPU

    env->fcr0 = MIPS_FCR0;	

#endif

    /* XXX some guesswork here, values are CPU specific */

    env->SYNCI_Step = 16;

    env->CCRes = 2;

}
