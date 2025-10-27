static int cpu_mips_register (CPUMIPSState *env, const mips_def_t *def)

{

    env->CP0_PRid = def->CP0_PRid;

    env->CP0_Config0 = def->CP0_Config0;

#ifdef TARGET_WORDS_BIGENDIAN

    env->CP0_Config0 |= (1 << CP0C0_BE);

#endif

    env->CP0_Config1 = def->CP0_Config1;

    env->CP0_Config2 = def->CP0_Config2;

    env->CP0_Config3 = def->CP0_Config3;

    env->CP0_Config6 = def->CP0_Config6;

    env->CP0_Config7 = def->CP0_Config7;

    env->SYNCI_Step = def->SYNCI_Step;

    env->CCRes = def->CCRes;

    env->CP0_Status_rw_bitmask = def->CP0_Status_rw_bitmask;

    env->CP0_TCStatus_rw_bitmask = def->CP0_TCStatus_rw_bitmask;

    env->CP0_SRSCtl = def->CP0_SRSCtl;

    env->current_tc = 0;

    env->SEGBITS = def->SEGBITS;

    env->SEGMask = (target_ulong)((1ULL << def->SEGBITS) - 1);

#if defined(TARGET_MIPS64)

    if (def->insn_flags & ISA_MIPS3) {

        env->hflags |= MIPS_HFLAG_64;

        env->SEGMask |= 3ULL << 62;

    }

#endif

    env->PABITS = def->PABITS;

    env->PAMask = (target_ulong)((1ULL << def->PABITS) - 1);

    env->CP0_SRSConf0_rw_bitmask = def->CP0_SRSConf0_rw_bitmask;

    env->CP0_SRSConf0 = def->CP0_SRSConf0;

    env->CP0_SRSConf1_rw_bitmask = def->CP0_SRSConf1_rw_bitmask;

    env->CP0_SRSConf1 = def->CP0_SRSConf1;

    env->CP0_SRSConf2_rw_bitmask = def->CP0_SRSConf2_rw_bitmask;

    env->CP0_SRSConf2 = def->CP0_SRSConf2;

    env->CP0_SRSConf3_rw_bitmask = def->CP0_SRSConf3_rw_bitmask;

    env->CP0_SRSConf3 = def->CP0_SRSConf3;

    env->CP0_SRSConf4_rw_bitmask = def->CP0_SRSConf4_rw_bitmask;

    env->CP0_SRSConf4 = def->CP0_SRSConf4;

    env->insn_flags = def->insn_flags;



#ifndef CONFIG_USER_ONLY

    if (!env->user_mode_only)

        mmu_init(env, def);

#endif

    fpu_init(env, def);

    mvp_init(env, def);

    return 0;

}
