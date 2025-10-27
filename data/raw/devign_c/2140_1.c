static int cpudef_setfield(const char *name, const char *str, void *opaque)

{

    x86_def_t *def = opaque;

    int err = 0;



    if (!strcmp(name, "name")) {


        def->name = g_strdup(str);

    } else if (!strcmp(name, "model_id")) {

        strncpy(def->model_id, str, sizeof (def->model_id));

    } else if (!strcmp(name, "level")) {

        setscalar(&def->level, str, &err)

    } else if (!strcmp(name, "vendor")) {

        cpyid(&str[0], &def->vendor1);

        cpyid(&str[4], &def->vendor2);

        cpyid(&str[8], &def->vendor3);

    } else if (!strcmp(name, "family")) {

        setscalar(&def->family, str, &err)

    } else if (!strcmp(name, "model")) {

        setscalar(&def->model, str, &err)

    } else if (!strcmp(name, "stepping")) {

        setscalar(&def->stepping, str, &err)

    } else if (!strcmp(name, "feature_edx")) {

        setfeatures(&def->features, str, feature_name, &err);

    } else if (!strcmp(name, "feature_ecx")) {

        setfeatures(&def->ext_features, str, ext_feature_name, &err);

    } else if (!strcmp(name, "extfeature_edx")) {

        setfeatures(&def->ext2_features, str, ext2_feature_name, &err);

    } else if (!strcmp(name, "extfeature_ecx")) {

        setfeatures(&def->ext3_features, str, ext3_feature_name, &err);

    } else if (!strcmp(name, "xlevel")) {

        setscalar(&def->xlevel, str, &err)

    } else {

        fprintf(stderr, "error: unknown option [%s = %s]\n", name, str);

        return (1);

    }

    if (err) {

        fprintf(stderr, "error: bad option value [%s = %s]\n", name, str);

        return (1);

    }

    return (0);

}