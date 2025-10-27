static int cpu_x86_register (CPUX86State *env, const char *cpu_model)

{

    x86_def_t def1, *def = &def1;



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

    env->cpuid_level = def->level;

    if (def->family > 0x0f)

        env->cpuid_version = 0xf00 | ((def->family - 0x0f) << 20);

    else

        env->cpuid_version = def->family << 8;

    env->cpuid_version |= ((def->model & 0xf) << 4) | ((def->model >> 4) << 16);

    env->cpuid_version |= def->stepping;

    env->cpuid_features = def->features;

    env->pat = 0x0007040600070406ULL;

    env->cpuid_ext_features = def->ext_features;

    env->cpuid_ext2_features = def->ext2_features;

    env->cpuid_xlevel = def->xlevel;

    env->cpuid_ext3_features = def->ext3_features;

    {

        const char *model_id = def->model_id;

        int c, len, i;

        if (!model_id)

            model_id = "";

        len = strlen(model_id);

        for(i = 0; i < 48; i++) {

            if (i >= len)

                c = '\0';

            else

                c = (uint8_t)model_id[i];

            env->cpuid_model[i >> 2] |= c << (8 * (i & 3));

        }

    }

    return 0;

}
