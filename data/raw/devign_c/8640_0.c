static void dacr_write(CPUARMState *env, const ARMCPRegInfo *ri, uint64_t value)

{

    ARMCPU *cpu = arm_env_get_cpu(env);



    env->cp15.c3 = value;

    tlb_flush(CPU(cpu), 1); /* Flush TLB as domain not tracked in TLB */

}
