static void init_excp_BookE (CPUPPCState *env)

{

#if !defined(CONFIG_USER_ONLY)

    env->excp_vectors[POWERPC_EXCP_CRITICAL] = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_MCHECK]   = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_DSI]      = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_ISI]      = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_EXTERNAL] = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_ALIGN]    = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_PROGRAM]  = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_FPU]      = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_SYSCALL]  = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_APU]      = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_DECR]     = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_FIT]      = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_WDT]      = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_DTLB]     = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_ITLB]     = 0x00000000;

    env->excp_vectors[POWERPC_EXCP_DEBUG]    = 0x00000000;

    env->excp_prefix = 0x00000000;

    env->ivor_mask = 0x0000FFE0;

    env->ivpr_mask = 0xFFFF0000;

    /* Hardware reset vector */

    env->hreset_vector = 0xFFFFFFFCUL;

#endif

}
