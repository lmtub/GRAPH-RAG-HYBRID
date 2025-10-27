static int cpu_sparc_find_by_name(sparc_def_t *cpu_def, const char *cpu_model)

{

    unsigned int i;

    const sparc_def_t *def = NULL;

    char *s = strdup(cpu_model);

    char *featurestr, *name = strtok(s, ",");

    uint32_t plus_features = 0;

    uint32_t minus_features = 0;

    uint64_t iu_version;

    uint32_t fpu_version, mmu_version, nwindows;



    for (i = 0; i < ARRAY_SIZE(sparc_defs); i++) {

        if (strcasecmp(name, sparc_defs[i].name) == 0) {

            def = &sparc_defs[i];

        }

    }

    if (!def) {

        goto error;

    }

    memcpy(cpu_def, def, sizeof(*def));



    featurestr = strtok(NULL, ",");

    while (featurestr) {

        char *val;



        if (featurestr[0] == '+') {

            add_flagname_to_bitmaps(featurestr + 1, &plus_features);

        } else if (featurestr[0] == '-') {

            add_flagname_to_bitmaps(featurestr + 1, &minus_features);

        } else if ((val = strchr(featurestr, '='))) {

            *val = 0; val++;

            if (!strcmp(featurestr, "iu_version")) {

                char *err;



                iu_version = strtoll(val, &err, 0);

                if (!*val || *err) {

                    fprintf(stderr, "bad numerical value %s\n", val);

                    goto error;

                }

                cpu_def->iu_version = iu_version;

#ifdef DEBUG_FEATURES

                fprintf(stderr, "iu_version %" PRIx64 "\n", iu_version);

#endif

            } else if (!strcmp(featurestr, "fpu_version")) {

                char *err;



                fpu_version = strtol(val, &err, 0);

                if (!*val || *err) {

                    fprintf(stderr, "bad numerical value %s\n", val);

                    goto error;

                }

                cpu_def->fpu_version = fpu_version;

#ifdef DEBUG_FEATURES

                fprintf(stderr, "fpu_version %x\n", fpu_version);

#endif

            } else if (!strcmp(featurestr, "mmu_version")) {

                char *err;



                mmu_version = strtol(val, &err, 0);

                if (!*val || *err) {

                    fprintf(stderr, "bad numerical value %s\n", val);

                    goto error;

                }

                cpu_def->mmu_version = mmu_version;

#ifdef DEBUG_FEATURES

                fprintf(stderr, "mmu_version %x\n", mmu_version);

#endif

            } else if (!strcmp(featurestr, "nwindows")) {

                char *err;



                nwindows = strtol(val, &err, 0);

                if (!*val || *err || nwindows > MAX_NWINDOWS ||

                    nwindows < MIN_NWINDOWS) {

                    fprintf(stderr, "bad numerical value %s\n", val);

                    goto error;

                }

                cpu_def->nwindows = nwindows;

#ifdef DEBUG_FEATURES

                fprintf(stderr, "nwindows %d\n", nwindows);

#endif

            } else {

                fprintf(stderr, "unrecognized feature %s\n", featurestr);

                goto error;

            }

        } else {

            fprintf(stderr, "feature string `%s' not in format "

                    "(+feature|-feature|feature=xyz)\n", featurestr);

            goto error;

        }

        featurestr = strtok(NULL, ",");

    }

    cpu_def->features |= plus_features;

    cpu_def->features &= ~minus_features;

#ifdef DEBUG_FEATURES

    print_features(stderr, fprintf, cpu_def->features, NULL);

#endif

    free(s);

    return 0;



 error:

    free(s);

    return -1;

}
