X86CPU *cpu_x86_create(const char *cpu_model, Error **errp)

{

    X86CPU *cpu = NULL;

    ObjectClass *oc;

    CPUClass *cc;

    gchar **model_pieces;

    char *name, *features;

    Error *error = NULL;

    const char *typename;



    model_pieces = g_strsplit(cpu_model, ",", 2);

    if (!model_pieces[0]) {

        error_setg(&error, "Invalid/empty CPU model name");

        goto out;

    }

    name = model_pieces[0];

    features = model_pieces[1];



    oc = x86_cpu_class_by_name(name);

    if (oc == NULL) {

        error_setg(&error, "Unable to find CPU definition: %s", name);

        goto out;

    }

    cc = CPU_CLASS(oc);

    typename = object_class_get_name(oc);



    cc->parse_features(typename, features, &error);

    cpu = X86_CPU(object_new(typename));

    if (error) {

        goto out;

    }



out:

    if (error != NULL) {

        error_propagate(errp, error);

        if (cpu) {

            object_unref(OBJECT(cpu));

            cpu = NULL;

        }

    }

    g_strfreev(model_pieces);

    return cpu;

}
