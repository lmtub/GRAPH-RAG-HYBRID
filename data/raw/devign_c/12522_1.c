static inline bool extended_addresses_enabled(CPUARMState *env)

{

    return arm_feature(env, ARM_FEATURE_LPAE)

        && (env->cp15.c2_control & (1 << 31));

}
