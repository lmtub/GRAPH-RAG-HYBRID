static int cpu_x86_find_by_name(x86_def_t *x86_cpu_def, const char *cpu_model)

{

    unsigned int i;

    x86_def_t *def;



    char *s = strdup(cpu_model);

    char *featurestr, *name = strtok(s, ",");

    uint32_t plus_features = 0, plus_ext_features = 0, plus_ext2_features = 0, plus_ext3_features = 0;

    uint32_t minus_features = 0, minus_ext_features = 0, minus_ext2_features = 0, minus_ext3_features = 0;

    int family = -1, model = -1, stepping = -1;



    def = NULL;

    for (i = 0; i < ARRAY_SIZE(x86_defs); i++) {

        if (strcmp(name, x86_defs[i].name) == 0) {

            def = &x86_defs[i];

            break;

        }

    }

    if (kvm_enabled() && strcmp(name, "host") == 0) {

        cpu_x86_fill_host(x86_cpu_def);

    } else if (!def) {

        goto error;

    } else {

        memcpy(x86_cpu_def, def, sizeof(*def));

    }



    add_flagname_to_bitmaps("hypervisor", &plus_features,

        &plus_ext_features, &plus_ext2_features, &plus_ext3_features);



    featurestr = strtok(NULL, ",");



    while (featurestr) {

        char *val;

        if (featurestr[0] == '+') {

            add_flagname_to_bitmaps(featurestr + 1, &plus_features, &plus_ext_features, &plus_ext2_features, &plus_ext3_features);

        } else if (featurestr[0] == '-') {

            add_flagname_to_bitmaps(featurestr + 1, &minus_features, &minus_ext_features, &minus_ext2_features, &minus_ext3_features);

        } else if ((val = strchr(featurestr, '='))) {

            *val = 0; val++;

            if (!strcmp(featurestr, "family")) {

                char *err;

                family = strtol(val, &err, 10);

                if (!*val || *err || family < 0) {

                    fprintf(stderr, "bad numerical value %s\n", val);

                    goto error;

                }

                x86_cpu_def->family = family;

            } else if (!strcmp(featurestr, "model")) {

                char *err;

                model = strtol(val, &err, 10);

                if (!*val || *err || model < 0 || model > 0xff) {

                    fprintf(stderr, "bad numerical value %s\n", val);

                    goto error;

                }

                x86_cpu_def->model = model;

            } else if (!strcmp(featurestr, "stepping")) {

                char *err;

                stepping = strtol(val, &err, 10);

                if (!*val || *err || stepping < 0 || stepping > 0xf) {

                    fprintf(stderr, "bad numerical value %s\n", val);

                    goto error;

                }

                x86_cpu_def->stepping = stepping;

            } else if (!strcmp(featurestr, "vendor")) {

                if (strlen(val) != 12) {

                    fprintf(stderr, "vendor string must be 12 chars long\n");

                    goto error;

                }

                x86_cpu_def->vendor1 = 0;

                x86_cpu_def->vendor2 = 0;

                x86_cpu_def->vendor3 = 0;

                for(i = 0; i < 4; i++) {

                    x86_cpu_def->vendor1 |= ((uint8_t)val[i    ]) << (8 * i);

                    x86_cpu_def->vendor2 |= ((uint8_t)val[i + 4]) << (8 * i);

                    x86_cpu_def->vendor3 |= ((uint8_t)val[i + 8]) << (8 * i);

                }

                x86_cpu_def->vendor_override = 1;

            } else if (!strcmp(featurestr, "model_id")) {

                pstrcpy(x86_cpu_def->model_id, sizeof(x86_cpu_def->model_id),

                        val);

            } else {

                fprintf(stderr, "unrecognized feature %s\n", featurestr);

                goto error;

            }

        } else {

            fprintf(stderr, "feature string `%s' not in format (+feature|-feature|feature=xyz)\n", featurestr);

            goto error;

        }

        featurestr = strtok(NULL, ",");

    }

    x86_cpu_def->features |= plus_features;

    x86_cpu_def->ext_features |= plus_ext_features;

    x86_cpu_def->ext2_features |= plus_ext2_features;

    x86_cpu_def->ext3_features |= plus_ext3_features;

    x86_cpu_def->features &= ~minus_features;

    x86_cpu_def->ext_features &= ~minus_ext_features;

    x86_cpu_def->ext2_features &= ~minus_ext2_features;

    x86_cpu_def->ext3_features &= ~minus_ext3_features;

    free(s);

    return 0;



error:

    free(s);

    return -1;

}
