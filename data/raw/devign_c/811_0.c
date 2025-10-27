static uint32_t get_elf_hwcap(void)

{

    ARMCPU *cpu = ARM_CPU(thread_cpu);

    uint32_t hwcaps = 0;



    hwcaps |= ARM_HWCAP_ARM_SWP;

    hwcaps |= ARM_HWCAP_ARM_HALF;

    hwcaps |= ARM_HWCAP_ARM_THUMB;

    hwcaps |= ARM_HWCAP_ARM_FAST_MULT;



    /* probe for the extra features */

#define GET_FEATURE(feat, hwcap) \

    do { if (arm_feature(&cpu->env, feat)) { hwcaps |= hwcap; } } while (0)

    /* EDSP is in v5TE and above, but all our v5 CPUs are v5TE */

    GET_FEATURE(ARM_FEATURE_V5, ARM_HWCAP_ARM_EDSP);

    GET_FEATURE(ARM_FEATURE_VFP, ARM_HWCAP_ARM_VFP);

    GET_FEATURE(ARM_FEATURE_IWMMXT, ARM_HWCAP_ARM_IWMMXT);

    GET_FEATURE(ARM_FEATURE_THUMB2EE, ARM_HWCAP_ARM_THUMBEE);

    GET_FEATURE(ARM_FEATURE_NEON, ARM_HWCAP_ARM_NEON);

    GET_FEATURE(ARM_FEATURE_VFP3, ARM_HWCAP_ARM_VFPv3);

    GET_FEATURE(ARM_FEATURE_V6K, ARM_HWCAP_ARM_TLS);

    GET_FEATURE(ARM_FEATURE_VFP4, ARM_HWCAP_ARM_VFPv4);

    GET_FEATURE(ARM_FEATURE_ARM_DIV, ARM_HWCAP_ARM_IDIVA);

    GET_FEATURE(ARM_FEATURE_THUMB_DIV, ARM_HWCAP_ARM_IDIVT);

    /* All QEMU's VFPv3 CPUs have 32 registers, see VFP_DREG in translate.c.

     * Note that the ARM_HWCAP_ARM_VFPv3D16 bit is always the inverse of

     * ARM_HWCAP_ARM_VFPD32 (and so always clear for QEMU); it is unrelated

     * to our VFP_FP16 feature bit.

     */

    GET_FEATURE(ARM_FEATURE_VFP3, ARM_HWCAP_ARM_VFPD32);

    GET_FEATURE(ARM_FEATURE_LPAE, ARM_HWCAP_ARM_LPAE);

#undef GET_FEATURE



    return hwcaps;

}
