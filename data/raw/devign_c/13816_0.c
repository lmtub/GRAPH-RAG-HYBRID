void kvm_mips_reset_vcpu(MIPSCPU *cpu)

{

    CPUMIPSState *env = &cpu->env;



    if (env->CP0_Config1 & (1 << CP0C1_FP)) {

        fprintf(stderr, "Warning: FPU not supported with KVM, disabling\n");

        env->CP0_Config1 &= ~(1 << CP0C1_FP);

    }



    DPRINTF("%s\n", __func__);

}
