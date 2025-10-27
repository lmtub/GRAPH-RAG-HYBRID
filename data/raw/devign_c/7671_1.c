static void x86_cpu_parse_featurestr(CPUState *cs, char *features,

                                     Error **errp)

{

    X86CPU *cpu = X86_CPU(cs);

    char *featurestr; /* Single 'key=value" string being parsed */

    /* Features to be added */

    FeatureWordArray plus_features = { 0 };

    /* Features to be removed */

    FeatureWordArray minus_features = { 0 };

    uint32_t numvalue;

    CPUX86State *env = &cpu->env;

    Error *local_err = NULL;



    featurestr = features ? strtok(features, ",") : NULL;



    while (featurestr) {

        char *val;

        if (featurestr[0] == '+') {

            add_flagname_to_bitmaps(featurestr + 1, plus_features);

        } else if (featurestr[0] == '-') {

            add_flagname_to_bitmaps(featurestr + 1, minus_features);

        } else if ((val = strchr(featurestr, '='))) {

            *val = 0; val++;

            feat2prop(featurestr);

            if (!strcmp(featurestr, "xlevel")) {

                char *err;

                char num[32];



                numvalue = strtoul(val, &err, 0);

                if (!*val || *err) {

                    error_setg(&local_err, "bad numerical value %s", val);

                    goto out;

                }

                if (numvalue < 0x80000000) {

                    error_report("xlevel value shall always be >= 0x80000000"

                                 ", fixup will be removed in future versions");

                    numvalue += 0x80000000;

                }

                snprintf(num, sizeof(num), "%" PRIu32, numvalue);

                object_property_parse(OBJECT(cpu), num, featurestr, &local_err);

            } else if (!strcmp(featurestr, "tsc-freq")) {

                int64_t tsc_freq;

                char *err;

                char num[32];



                tsc_freq = strtosz_suffix_unit(val, &err,

                                               STRTOSZ_DEFSUFFIX_B, 1000);

                if (tsc_freq < 0 || *err) {

                    error_setg(&local_err, "bad numerical value %s", val);

                    goto out;

                }

                snprintf(num, sizeof(num), "%" PRId64, tsc_freq);

                object_property_parse(OBJECT(cpu), num, "tsc-frequency",

                                      &local_err);

            } else if (!strcmp(featurestr, "hv-spinlocks")) {

                char *err;

                const int min = 0xFFF;

                char num[32];

                numvalue = strtoul(val, &err, 0);

                if (!*val || *err) {

                    error_setg(&local_err, "bad numerical value %s", val);

                    goto out;

                }

                if (numvalue < min) {

                    error_report("hv-spinlocks value shall always be >= 0x%x"

                            ", fixup will be removed in future versions",

                            min);

                    numvalue = min;

                }

                snprintf(num, sizeof(num), "%" PRId32, numvalue);

                object_property_parse(OBJECT(cpu), num, featurestr, &local_err);

            } else {

                object_property_parse(OBJECT(cpu), val, featurestr, &local_err);

            }

        } else {

            feat2prop(featurestr);

            object_property_parse(OBJECT(cpu), "on", featurestr, &local_err);

        }

        if (local_err) {

            error_propagate(errp, local_err);

            goto out;

        }

        featurestr = strtok(NULL, ",");

    }

    env->features[FEAT_1_EDX] |= plus_features[FEAT_1_EDX];

    env->features[FEAT_1_ECX] |= plus_features[FEAT_1_ECX];

    env->features[FEAT_8000_0001_EDX] |= plus_features[FEAT_8000_0001_EDX];

    env->features[FEAT_8000_0001_ECX] |= plus_features[FEAT_8000_0001_ECX];

    env->features[FEAT_C000_0001_EDX] |= plus_features[FEAT_C000_0001_EDX];

    env->features[FEAT_KVM] |= plus_features[FEAT_KVM];

    env->features[FEAT_SVM] |= plus_features[FEAT_SVM];

    env->features[FEAT_7_0_EBX] |= plus_features[FEAT_7_0_EBX];

    env->features[FEAT_1_EDX] &= ~minus_features[FEAT_1_EDX];

    env->features[FEAT_1_ECX] &= ~minus_features[FEAT_1_ECX];

    env->features[FEAT_8000_0001_EDX] &= ~minus_features[FEAT_8000_0001_EDX];

    env->features[FEAT_8000_0001_ECX] &= ~minus_features[FEAT_8000_0001_ECX];

    env->features[FEAT_C000_0001_EDX] &= ~minus_features[FEAT_C000_0001_EDX];

    env->features[FEAT_KVM] &= ~minus_features[FEAT_KVM];

    env->features[FEAT_SVM] &= ~minus_features[FEAT_SVM];

    env->features[FEAT_7_0_EBX] &= ~minus_features[FEAT_7_0_EBX];



out:

    return;

}
