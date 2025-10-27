static void contextidr_write(CPUARMState *env, const ARMCPRegInfo *ri,

                             uint64_t value)

{

    ARMCPU *cpu = arm_env_get_cpu(env);



    if (env->cp15.contextidr_el1 != value && !arm_feature(env, ARM_FEATURE_MPU)

        && !extended_addresses_enabled(env)) {

        /* For VMSA (when not using the LPAE long descriptor page table

         * format) this register includes the ASID, so do a TLB flush.

         * For PMSA it is purely a process ID and no action is needed.

         */

        tlb_flush(CPU(cpu), 1);

    }

    env->cp15.contextidr_el1 = value;

}
