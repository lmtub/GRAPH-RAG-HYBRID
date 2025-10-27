static inline void do_rfi(CPUPPCState *env, target_ulong nip, target_ulong msr)

{

    CPUState *cs = CPU(ppc_env_get_cpu(env));



    /* MSR:POW cannot be set by any form of rfi */

    msr &= ~(1ULL << MSR_POW);



#if defined(TARGET_PPC64)

    /* Switching to 32-bit ? Crop the nip */

    if (!msr_is_64bit(env, msr)) {

        nip = (uint32_t)nip;

    }

#else

    nip = (uint32_t)nip;

#endif

    /* XXX: beware: this is false if VLE is supported */

    env->nip = nip & ~((target_ulong)0x00000003);

    hreg_store_msr(env, msr, 1);

#if defined(DEBUG_OP)

    cpu_dump_rfi(env->nip, env->msr);

#endif

    /* No need to raise an exception here,

     * as rfi is always the last insn of a TB

     */

    cs->interrupt_request |= CPU_INTERRUPT_EXITTB;



    /* Context synchronizing: check if TCG TLB needs flush */

    check_tlb_flush(env);

}
