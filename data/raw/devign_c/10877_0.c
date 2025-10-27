void cpu_x86_update_cr4(CPUX86State *env, uint32_t new_cr4)

{

#if defined(DEBUG_MMU)

    printf("CR4 update: CR4=%08x\n", (uint32_t)env->cr[4]);

#endif

    if ((new_cr4 & (CR4_PGE_MASK | CR4_PAE_MASK | CR4_PSE_MASK)) !=

        (env->cr[4] & (CR4_PGE_MASK | CR4_PAE_MASK | CR4_PSE_MASK))) {

        tlb_flush(env, 1);

    }

    /* SSE handling */

    if (!(env->cpuid_features & CPUID_SSE))

        new_cr4 &= ~CR4_OSFXSR_MASK;

    if (new_cr4 & CR4_OSFXSR_MASK)

        env->hflags |= HF_OSFXSR_MASK;

    else

        env->hflags &= ~HF_OSFXSR_MASK;



    env->cr[4] = new_cr4;

}
