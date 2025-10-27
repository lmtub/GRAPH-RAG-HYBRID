static int vmsa_ttbcr_raw_write(CPUARMState *env, const ARMCPRegInfo *ri,

                                uint64_t value)

{

    if (arm_feature(env, ARM_FEATURE_LPAE)) {

        value &= ~((7 << 19) | (3 << 14) | (0xf << 3));

    } else {

        value &= 7;

    }

    /* Note that we always calculate c2_mask and c2_base_mask, but

     * they are only used for short-descriptor tables (ie if EAE is 0);

     * for long-descriptor tables the TTBCR fields are used differently

     * and the c2_mask and c2_base_mask values are meaningless.

     */

    env->cp15.c2_control = value;

    env->cp15.c2_mask = ~(((uint32_t)0xffffffffu) >> value);

    env->cp15.c2_base_mask = ~((uint32_t)0x3fffu >> value);

    return 0;

}
