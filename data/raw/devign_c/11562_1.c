static uint64_t pmsav5_data_ap_read(CPUARMState *env, const ARMCPRegInfo *ri)

{

    return simple_mpu_ap_bits(env->cp15.c5_data);

}
