static bool pmsav7_use_background_region(ARMCPU *cpu,

                                         ARMMMUIdx mmu_idx, bool is_user)

{

    /* Return true if we should use the default memory map as a

     * "background" region if there are no hits against any MPU regions.

     */

    CPUARMState *env = &cpu->env;



    if (is_user) {

        return false;

    }



    if (arm_feature(env, ARM_FEATURE_M)) {

        return env->v7m.mpu_ctrl & R_V7M_MPU_CTRL_PRIVDEFENA_MASK;

    } else {

        return regime_sctlr(env, mmu_idx) & SCTLR_BR;

    }

}
