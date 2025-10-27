static void x86_cpu_register_feature_bit_props(X86CPU *cpu,

                                               FeatureWord w,

                                               int bitnr)

{

    Object *obj = OBJECT(cpu);

    int i;

    char **names;

    FeatureWordInfo *fi = &feature_word_info[w];



    if (!fi->feat_names[bitnr]) {

        return;

    }



    names = g_strsplit(fi->feat_names[bitnr], "|", 0);



    feat2prop(names[0]);

    x86_cpu_register_bit_prop(cpu, names[0], &cpu->env.features[w], bitnr);



    for (i = 1; names[i]; i++) {

        feat2prop(names[i]);

        object_property_add_alias(obj, names[i], obj, names[0],

                                  &error_abort);

    }



    g_strfreev(names);

}
