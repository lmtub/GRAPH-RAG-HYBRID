CPUPPCState *cpu_ppc_init(void)

{

    CPUPPCState *env;



    cpu_exec_init();



    env = qemu_mallocz(sizeof(CPUPPCState));

    if (!env)

        return NULL;

#if !defined(CONFIG_USER_ONLY) && defined (USE_OPEN_FIRMWARE)

    setup_machine(env, 0);

#else

//    env->spr[PVR] = 0; /* Basic PPC */

    env->spr[PVR] = 0x00080100; /* G3 CPU */

//    env->spr[PVR] = 0x00083100; /* MPC755 (G3 embedded) */

//    env->spr[PVR] = 0x00070100; /* IBM 750FX */

#endif

    tlb_flush(env, 1);

#if defined (DO_SINGLE_STEP)

    /* Single step trace mode */

    msr_se = 1;

#endif

    msr_fp = 1; /* Allow floating point exceptions */

    msr_me = 1; /* Allow machine check exceptions  */

#if defined(CONFIG_USER_ONLY)

    msr_pr = 1;

    cpu_ppc_register(env, 0x00080000);

#else

    env->nip = 0xFFFFFFFC;

#endif

    env->access_type = ACCESS_INT;

    cpu_single_env = env;

    return env;

}
