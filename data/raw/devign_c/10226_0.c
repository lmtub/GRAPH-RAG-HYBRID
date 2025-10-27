static void vmsa_ttbcr_reset(CPUARMState *env, const ARMCPRegInfo *ri)

{

    env->cp15.c2_base_mask = 0xffffc000u;

    env->cp15.c2_control = 0;

    env->cp15.c2_mask = 0;

}
