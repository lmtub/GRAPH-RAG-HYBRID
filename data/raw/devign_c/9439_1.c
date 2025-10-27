static void arm_cpu_realizefn(DeviceState *dev, Error **errp)

{

    CPUState *cs = CPU(dev);

    ARMCPU *cpu = ARM_CPU(dev);

    ARMCPUClass *acc = ARM_CPU_GET_CLASS(dev);

    CPUARMState *env = &cpu->env;

    int pagebits;

    Error *local_err = NULL;



    cpu_exec_realizefn(cs, &local_err);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        return;

    }



    /* Some features automatically imply others: */

    if (arm_feature(env, ARM_FEATURE_V8)) {

        set_feature(env, ARM_FEATURE_V7);

        set_feature(env, ARM_FEATURE_ARM_DIV);

        set_feature(env, ARM_FEATURE_LPAE);

    }

    if (arm_feature(env, ARM_FEATURE_V7)) {

        set_feature(env, ARM_FEATURE_VAPA);

        set_feature(env, ARM_FEATURE_THUMB2);

        set_feature(env, ARM_FEATURE_MPIDR);

        if (!arm_feature(env, ARM_FEATURE_M)) {

            set_feature(env, ARM_FEATURE_V6K);

        } else {

            set_feature(env, ARM_FEATURE_V6);

        }



        /* Always define VBAR for V7 CPUs even if it doesn't exist in

         * non-EL3 configs. This is needed by some legacy boards.

         */

        set_feature(env, ARM_FEATURE_VBAR);

    }

    if (arm_feature(env, ARM_FEATURE_V6K)) {

        set_feature(env, ARM_FEATURE_V6);

        set_feature(env, ARM_FEATURE_MVFR);

    }

    if (arm_feature(env, ARM_FEATURE_V6)) {

        set_feature(env, ARM_FEATURE_V5);


        if (!arm_feature(env, ARM_FEATURE_M)) {

            set_feature(env, ARM_FEATURE_AUXCR);

        }

    }

    if (arm_feature(env, ARM_FEATURE_V5)) {

        set_feature(env, ARM_FEATURE_V4T);

    }

    if (arm_feature(env, ARM_FEATURE_M)) {

        set_feature(env, ARM_FEATURE_THUMB_DIV);

    }

    if (arm_feature(env, ARM_FEATURE_ARM_DIV)) {

        set_feature(env, ARM_FEATURE_THUMB_DIV);

    }

    if (arm_feature(env, ARM_FEATURE_VFP4)) {

        set_feature(env, ARM_FEATURE_VFP3);

        set_feature(env, ARM_FEATURE_VFP_FP16);

    }

    if (arm_feature(env, ARM_FEATURE_VFP3)) {

        set_feature(env, ARM_FEATURE_VFP);

    }

    if (arm_feature(env, ARM_FEATURE_LPAE)) {

        set_feature(env, ARM_FEATURE_V7MP);

        set_feature(env, ARM_FEATURE_PXN);

    }

    if (arm_feature(env, ARM_FEATURE_CBAR_RO)) {

        set_feature(env, ARM_FEATURE_CBAR);

    }

    if (arm_feature(env, ARM_FEATURE_THUMB2) &&

        !arm_feature(env, ARM_FEATURE_M)) {

        set_feature(env, ARM_FEATURE_THUMB_DSP);

    }



    if (arm_feature(env, ARM_FEATURE_V7) &&

        !arm_feature(env, ARM_FEATURE_M) &&

        !arm_feature(env, ARM_FEATURE_PMSA)) {

        /* v7VMSA drops support for the old ARMv5 tiny pages, so we

         * can use 4K pages.

         */

        pagebits = 12;

    } else {

        /* For CPUs which might have tiny 1K pages, or which have an

         * MPU and might have small region sizes, stick with 1K pages.

         */

        pagebits = 10;

    }

    if (!set_preferred_target_page_bits(pagebits)) {

        /* This can only ever happen for hotplugging a CPU, or if

         * the board code incorrectly creates a CPU which it has

         * promised via minimum_page_size that it will not.

         */

        error_setg(errp, "This CPU requires a smaller page size than the "

                   "system is using");

        return;

    }



    /* This cpu-id-to-MPIDR affinity is used only for TCG; KVM will override it.

     * We don't support setting cluster ID ([16..23]) (known as Aff2

     * in later ARM ARM versions), or any of the higher affinity level fields,

     * so these bits always RAZ.

     */

    if (cpu->mp_affinity == ARM64_AFFINITY_INVALID) {

        cpu->mp_affinity = arm_cpu_mp_affinity(cs->cpu_index,

                                               ARM_DEFAULT_CPUS_PER_CLUSTER);

    }



    if (cpu->reset_hivecs) {

            cpu->reset_sctlr |= (1 << 13);

    }



    if (cpu->cfgend) {

        if (arm_feature(&cpu->env, ARM_FEATURE_V7)) {

            cpu->reset_sctlr |= SCTLR_EE;

        } else {

            cpu->reset_sctlr |= SCTLR_B;

        }

    }



    if (!cpu->has_el3) {

        /* If the has_el3 CPU property is disabled then we need to disable the

         * feature.

         */

        unset_feature(env, ARM_FEATURE_EL3);



        /* Disable the security extension feature bits in the processor feature

         * registers as well. These are id_pfr1[7:4] and id_aa64pfr0[15:12].

         */

        cpu->id_pfr1 &= ~0xf0;

        cpu->id_aa64pfr0 &= ~0xf000;

    }



    if (!cpu->has_el2) {

        unset_feature(env, ARM_FEATURE_EL2);

    }



    if (!cpu->has_pmu) {

        unset_feature(env, ARM_FEATURE_PMU);

        cpu->id_aa64dfr0 &= ~0xf00;

    }



    if (!arm_feature(env, ARM_FEATURE_EL2)) {

        /* Disable the hypervisor feature bits in the processor feature

         * registers if we don't have EL2. These are id_pfr1[15:12] and

         * id_aa64pfr0_el1[11:8].

         */

        cpu->id_aa64pfr0 &= ~0xf00;

        cpu->id_pfr1 &= ~0xf000;

    }



    /* MPU can be configured out of a PMSA CPU either by setting has-mpu

     * to false or by setting pmsav7-dregion to 0.

     */

    if (!cpu->has_mpu) {

        cpu->pmsav7_dregion = 0;

    }

    if (cpu->pmsav7_dregion == 0) {

        cpu->has_mpu = false;

    }



    if (arm_feature(env, ARM_FEATURE_PMSA) &&

        arm_feature(env, ARM_FEATURE_V7)) {

        uint32_t nr = cpu->pmsav7_dregion;



        if (nr > 0xff) {

            error_setg(errp, "PMSAv7 MPU #regions invalid %" PRIu32, nr);

            return;

        }



        if (nr) {

            if (arm_feature(env, ARM_FEATURE_V8)) {

                /* PMSAv8 */

                env->pmsav8.rbar[M_REG_NS] = g_new0(uint32_t, nr);

                env->pmsav8.rlar[M_REG_NS] = g_new0(uint32_t, nr);

                if (arm_feature(env, ARM_FEATURE_M_SECURITY)) {

                    env->pmsav8.rbar[M_REG_S] = g_new0(uint32_t, nr);

                    env->pmsav8.rlar[M_REG_S] = g_new0(uint32_t, nr);

                }

            } else {

                env->pmsav7.drbar = g_new0(uint32_t, nr);

                env->pmsav7.drsr = g_new0(uint32_t, nr);

                env->pmsav7.dracr = g_new0(uint32_t, nr);

            }

        }

    }



    if (arm_feature(env, ARM_FEATURE_EL3)) {

        set_feature(env, ARM_FEATURE_VBAR);

    }



    register_cp_regs_for_features(cpu);

    arm_cpu_register_gdb_regs_for_features(cpu);



    init_cpreg_list(cpu);



#ifndef CONFIG_USER_ONLY

    if (cpu->has_el3 || arm_feature(env, ARM_FEATURE_M_SECURITY)) {

        AddressSpace *as;



        cs->num_ases = 2;



        if (!cpu->secure_memory) {

            cpu->secure_memory = cs->memory;

        }

        as = address_space_init_shareable(cpu->secure_memory,

                                          "cpu-secure-memory");

        cpu_address_space_init(cs, as, ARMASIdx_S);

    } else {

        cs->num_ases = 1;

    }



    cpu_address_space_init(cs,

                           address_space_init_shareable(cs->memory,

                                                        "cpu-memory"),

                           ARMASIdx_NS);

#endif



    qemu_init_vcpu(cs);

    cpu_reset(cs);



    acc->parent_realize(dev, errp);

}