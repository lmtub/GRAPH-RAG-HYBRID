static void init_excp_970 (CPUPPCState *env)

{

#if !defined(CONFIG_USER_ONLY)

    env->excp_vectors[POWERPC_EXCP_RESET]    = 0x00000100;

    env->excp_vectors[POWERPC_EXCP_MCHECK]   = 0x00000200;

    env->excp_vectors[POWERPC_EXCP_DSI]      = 0x00000300;

    env->excp_vectors[POWERPC_EXCP_DSEG]     = 0x00000380;

    env->excp_vectors[POWERPC_EXCP_ISI]      = 0x00000400;

    env->excp_vectors[POWERPC_EXCP_ISEG]     = 0x00000480;

    env->excp_vectors[POWERPC_EXCP_EXTERNAL] = 0x00000500;

    env->excp_vectors[POWERPC_EXCP_ALIGN]    = 0x00000600;

    env->excp_vectors[POWERPC_EXCP_PROGRAM]  = 0x00000700;

    env->excp_vectors[POWERPC_EXCP_FPU]      = 0x00000800;

    env->excp_vectors[POWERPC_EXCP_DECR]     = 0x00000900;

#if defined(TARGET_PPC64H) /* PowerPC 64 with hypervisor mode support */

    env->excp_vectors[POWERPC_EXCP_HDECR]    = 0x00000980;

#endif

    env->excp_vectors[POWERPC_EXCP_SYSCALL]  = 0x00000C00;

    env->excp_vectors[POWERPC_EXCP_TRACE]    = 0x00000D00;

    env->excp_vectors[POWERPC_EXCP_PERFM]    = 0x00000F00;

    env->excp_vectors[POWERPC_EXCP_VPU]      = 0x00000F20;

    env->excp_vectors[POWERPC_EXCP_IABR]     = 0x00001300;

    env->excp_vectors[POWERPC_EXCP_MAINT]    = 0x00001600;

    env->excp_vectors[POWERPC_EXCP_VPUA]     = 0x00001700;

    env->excp_vectors[POWERPC_EXCP_THERM]    = 0x00001800;

    env->excp_prefix   = 0x00000000FFF00000ULL;

    /* Hardware reset vector */

    env->hreset_vector = 0x0000000000000100ULL;

#endif

}
