bool arm_regime_using_lpae_format(CPUARMState *env, ARMMMUIdx mmu_idx)

{

    return regime_using_lpae_format(env, mmu_idx);

}
