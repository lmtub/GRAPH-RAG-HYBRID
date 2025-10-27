static int kvm_mips_get_fpu_registers(CPUState *cs)

{

    MIPSCPU *cpu = MIPS_CPU(cs);

    CPUMIPSState *env = &cpu->env;

    int err, ret = 0;

    unsigned int i;



    /* Only get FPU state if we're emulating a CPU with an FPU */

    if (env->CP0_Config1 & (1 << CP0C1_FP)) {

        /* FPU Control Registers */

        err = kvm_mips_get_one_ureg(cs, KVM_REG_MIPS_FCR_IR,

                                    &env->active_fpu.fcr0);

        if (err < 0) {

            DPRINTF("%s: Failed to get FCR_IR (%d)\n", __func__, err);

            ret = err;

        }

        err = kvm_mips_get_one_ureg(cs, KVM_REG_MIPS_FCR_CSR,

                                    &env->active_fpu.fcr31);

        if (err < 0) {

            DPRINTF("%s: Failed to get FCR_CSR (%d)\n", __func__, err);

            ret = err;

        } else {

            restore_fp_status(env);

        }



        /* Floating point registers */

        for (i = 0; i < 32; ++i) {

            if (env->CP0_Status & (1 << CP0St_FR)) {

                err = kvm_mips_get_one_ureg64(cs, KVM_REG_MIPS_FPR_64(i),

                                              &env->active_fpu.fpr[i].d);

            } else {

                err = kvm_mips_get_one_ureg(cs, KVM_REG_MIPS_FPR_32(i),

                                      &env->active_fpu.fpr[i].w[FP_ENDIAN_IDX]);

            }

            if (err < 0) {

                DPRINTF("%s: Failed to get FPR%u (%d)\n", __func__, i, err);

                ret = err;

            }

        }

    }



    return ret;

}
