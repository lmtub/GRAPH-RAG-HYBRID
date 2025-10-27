static void x86_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    X86CPU *cpu = X86_CPU(obj);

    X86CPUClass *xcc = X86_CPU_GET_CLASS(obj);

    CPUX86State *env = &cpu->env;

    FeatureWord w;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    object_property_add(obj, "family", "int",

                        x86_cpuid_version_get_family,

                        x86_cpuid_version_set_family, NULL, NULL, NULL);

    object_property_add(obj, "model", "int",

                        x86_cpuid_version_get_model,

                        x86_cpuid_version_set_model, NULL, NULL, NULL);

    object_property_add(obj, "stepping", "int",

                        x86_cpuid_version_get_stepping,

                        x86_cpuid_version_set_stepping, NULL, NULL, NULL);

    object_property_add_str(obj, "vendor",

                            x86_cpuid_get_vendor,

                            x86_cpuid_set_vendor, NULL);

    object_property_add_str(obj, "model-id",

                            x86_cpuid_get_model_id,

                            x86_cpuid_set_model_id, NULL);

    object_property_add(obj, "tsc-frequency", "int",

                        x86_cpuid_get_tsc_freq,

                        x86_cpuid_set_tsc_freq, NULL, NULL, NULL);

    object_property_add(obj, "apic-id", "int",

                        x86_cpuid_get_apic_id,

                        x86_cpuid_set_apic_id, NULL, NULL, NULL);

    object_property_add(obj, "feature-words", "X86CPUFeatureWordInfo",

                        x86_cpu_get_feature_words,

                        NULL, NULL, (void *)env->features, NULL);

    object_property_add(obj, "filtered-features", "X86CPUFeatureWordInfo",

                        x86_cpu_get_feature_words,

                        NULL, NULL, (void *)cpu->filtered_features, NULL);



    cpu->hyperv_spinlock_attempts = HYPERV_SPINLOCK_NEVER_RETRY;



#ifndef CONFIG_USER_ONLY

    /* Any code creating new X86CPU objects have to set apic-id explicitly */

    cpu->apic_id = -1;

#endif



    for (w = 0; w < FEATURE_WORDS; w++) {

        int bitnr;



        for (bitnr = 0; bitnr < 32; bitnr++) {

            x86_cpu_register_feature_bit_props(cpu, w, bitnr);

        }

    }



    x86_cpu_load_def(cpu, xcc->cpu_def, &error_abort);

}
