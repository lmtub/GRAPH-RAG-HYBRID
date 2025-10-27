static void x86_cpu_load_features(X86CPU *cpu, Error **errp)

{

    CPUX86State *env = &cpu->env;

    FeatureWord w;

    GList *l;

    Error *local_err = NULL;



    /*TODO: cpu->max_features incorrectly overwrites features

     * set using "feat=on|off". Once we fix this, we can convert

     * plus_features & minus_features to global properties

     * inside x86_cpu_parse_featurestr() too.

     */

    if (cpu->max_features) {

        for (w = 0; w < FEATURE_WORDS; w++) {

            env->features[w] =

                x86_cpu_get_supported_feature_word(w, cpu->migratable);

        }

    }



    for (l = plus_features; l; l = l->next) {

        const char *prop = l->data;

        object_property_set_bool(OBJECT(cpu), true, prop, &local_err);

        if (local_err) {

            goto out;

        }

    }



    for (l = minus_features; l; l = l->next) {

        const char *prop = l->data;

        object_property_set_bool(OBJECT(cpu), false, prop, &local_err);

        if (local_err) {

            goto out;

        }

    }



    if (!kvm_enabled() || !cpu->expose_kvm) {

        env->features[FEAT_KVM] = 0;

    }



    x86_cpu_enable_xsave_components(cpu);



    /* CPUID[EAX=7,ECX=0].EBX always increased level automatically: */

    x86_cpu_adjust_feat_level(cpu, FEAT_7_0_EBX);

    if (cpu->full_cpuid_auto_level) {

        x86_cpu_adjust_feat_level(cpu, FEAT_1_EDX);

        x86_cpu_adjust_feat_level(cpu, FEAT_1_ECX);

        x86_cpu_adjust_feat_level(cpu, FEAT_6_EAX);

        x86_cpu_adjust_feat_level(cpu, FEAT_7_0_ECX);

        x86_cpu_adjust_feat_level(cpu, FEAT_8000_0001_EDX);

        x86_cpu_adjust_feat_level(cpu, FEAT_8000_0001_ECX);

        x86_cpu_adjust_feat_level(cpu, FEAT_8000_0007_EDX);

        x86_cpu_adjust_feat_level(cpu, FEAT_C000_0001_EDX);

        x86_cpu_adjust_feat_level(cpu, FEAT_SVM);

        x86_cpu_adjust_feat_level(cpu, FEAT_XSAVE);

        /* SVM requires CPUID[0x8000000A] */

        if (env->features[FEAT_8000_0001_ECX] & CPUID_EXT3_SVM) {

            x86_cpu_adjust_level(cpu, &env->cpuid_min_xlevel, 0x8000000A);

        }

    }



    /* Set cpuid_*level* based on cpuid_min_*level, if not explicitly set */

    if (env->cpuid_level == UINT32_MAX) {

        env->cpuid_level = env->cpuid_min_level;

    }

    if (env->cpuid_xlevel == UINT32_MAX) {

        env->cpuid_xlevel = env->cpuid_min_xlevel;

    }

    if (env->cpuid_xlevel2 == UINT32_MAX) {

        env->cpuid_xlevel2 = env->cpuid_min_xlevel2;

    }



out:

    if (local_err != NULL) {

        error_propagate(errp, local_err);

    }

}
