static void vmsa_tcr_el1_write(CPUARMState *env, const ARMCPRegInfo *ri,

                               uint64_t value)

{

    ARMCPU *cpu = arm_env_get_cpu(env);



    /* For AArch64 the A1 bit could result in a change of ASID, so TLB flush. */

    tlb_flush(CPU(cpu), 1);

    env->cp15.c2_control = value;

}
