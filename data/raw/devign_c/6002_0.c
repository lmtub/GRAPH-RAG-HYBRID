static void x86_cpu_parse_featurestr(CPUState *cs, char *features,

                                     Error **errp)

{

    X86CPU *cpu = X86_CPU(cs);

    char *featurestr; /* Single 'key=value" string being parsed */

    FeatureWord w;

    /* Features to be added */

    FeatureWordArray plus_features = { 0 };

    /* Features to be removed */

    FeatureWordArray minus_features = { 0 };

    CPUX86State *env = &cpu->env;

    Error *local_err = NULL;



    featurestr = features ? strtok(features, ",") : NULL;



    while (featurestr) {

        char *val;

        if (featurestr[0] == '+') {

            add_flagname_to_bitmaps(featurestr + 1, plus_features, &local_err);

        } else if (featurestr[0] == '-') {

            add_flagname_to_bitmaps(featurestr + 1, minus_features, &local_err);

        } else if ((val = strchr(featurestr, '='))) {

            *val = 0; val++;

            feat2prop(featurestr);

            if (!strcmp(featurestr, "tsc-freq")) {

                int64_t tsc_freq;

                char *err;

                char num[32];



                tsc_freq = qemu_strtosz_suffix_unit(val, &err,

                                               QEMU_STRTOSZ_DEFSUFFIX_B, 1000);

                if (tsc_freq < 0 || *err) {

                    error_setg(errp, "bad numerical value %s", val);

                    return;

                }

                snprintf(num, sizeof(num), "%" PRId64, tsc_freq);

                object_property_parse(OBJECT(cpu), num, "tsc-frequency",

                                      &local_err);

            } else {

                object_property_parse(OBJECT(cpu), val, featurestr, &local_err);

            }

        } else {

            feat2prop(featurestr);

            object_property_parse(OBJECT(cpu), "on", featurestr, &local_err);

        }

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

        featurestr = strtok(NULL, ",");

    }



    if (cpu->host_features) {

        for (w = 0; w < FEATURE_WORDS; w++) {

            env->features[w] =

                x86_cpu_get_supported_feature_word(w, cpu->migratable);

        }

    }



    for (w = 0; w < FEATURE_WORDS; w++) {

        env->features[w] |= plus_features[w];

        env->features[w] &= ~minus_features[w];

    }

}
