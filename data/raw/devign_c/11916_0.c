int cpu_x86_register(X86CPU *cpu, const char *cpu_model)

{

    CPUX86State *env = &cpu->env;

    x86_def_t def1, *def = &def1;

    Error *error = NULL;



    memset(def, 0, sizeof(*def));



    if (cpu_x86_find_by_name(def, cpu_model) < 0)

        return -1;

    if (def->vendor1) {

        env->cpuid_vendor1 = def->vendor1;

        env->cpuid_vendor2 = def->vendor2;

        env->cpuid_vendor3 = def->vendor3;

    } else {

        env->cpuid_vendor1 = CPUID_VENDOR_INTEL_1;

        env->cpuid_vendor2 = CPUID_VENDOR_INTEL_2;

        env->cpuid_vendor3 = CPUID_VENDOR_INTEL_3;

    }

    env->cpuid_vendor_override = def->vendor_override;

    object_property_set_int(OBJECT(cpu), def->level, "level", &error);

    object_property_set_int(OBJECT(cpu), def->family, "family", &error);

    object_property_set_int(OBJECT(cpu), def->model, "model", &error);

    object_property_set_int(OBJECT(cpu), def->stepping, "stepping", &error);

    env->cpuid_features = def->features;

    env->cpuid_ext_features = def->ext_features;

    env->cpuid_ext2_features = def->ext2_features;

    env->cpuid_ext3_features = def->ext3_features;

    object_property_set_int(OBJECT(cpu), def->xlevel, "xlevel", &error);

    env->cpuid_kvm_features = def->kvm_features;

    env->cpuid_svm_features = def->svm_features;

    env->cpuid_ext4_features = def->ext4_features;

    env->cpuid_7_0_ebx = def->cpuid_7_0_ebx_features;

    env->cpuid_xlevel2 = def->xlevel2;

    object_property_set_int(OBJECT(cpu), (int64_t)def->tsc_khz * 1000,

                            "tsc-frequency", &error);



    /* On AMD CPUs, some CPUID[8000_0001].EDX bits must match the bits on

     * CPUID[1].EDX.

     */

    if (env->cpuid_vendor1 == CPUID_VENDOR_AMD_1 &&

            env->cpuid_vendor2 == CPUID_VENDOR_AMD_2 &&

            env->cpuid_vendor3 == CPUID_VENDOR_AMD_3) {

        env->cpuid_ext2_features &= ~CPUID_EXT2_AMD_ALIASES;

        env->cpuid_ext2_features |= (def->features & CPUID_EXT2_AMD_ALIASES);

    }



    if (!kvm_enabled()) {

        env->cpuid_features &= TCG_FEATURES;

        env->cpuid_ext_features &= TCG_EXT_FEATURES;

        env->cpuid_ext2_features &= (TCG_EXT2_FEATURES

#ifdef TARGET_X86_64

            | CPUID_EXT2_SYSCALL | CPUID_EXT2_LM

#endif

            );

        env->cpuid_ext3_features &= TCG_EXT3_FEATURES;

        env->cpuid_svm_features &= TCG_SVM_FEATURES;

    }

    object_property_set_str(OBJECT(cpu), def->model_id, "model-id", &error);

    if (error_is_set(&error)) {

        error_free(error);

        return -1;

    }

    return 0;

}
