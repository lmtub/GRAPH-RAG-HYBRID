static void fcse_write(CPUARMState *env, const ARMCPRegInfo *ri, uint64_t value)

{

    ARMCPU *cpu = arm_env_get_cpu(env);



    if (env->cp15.c13_fcse != value) {

        /* Unlike real hardware the qemu TLB uses virtual addresses,

         * not modified virtual addresses, so this causes a TLB flush.

         */

        tlb_flush(CPU(cpu), 1);

        env->cp15.c13_fcse = value;

    }

}
