static void create_cpu_model_list(ObjectClass *klass, void *opaque)

{

    struct CpuDefinitionInfoListData *cpu_list_data = opaque;

    CpuDefinitionInfoList **cpu_list = &cpu_list_data->list;

    CpuDefinitionInfoList *entry;

    CpuDefinitionInfo *info;

    char *name = g_strdup(object_class_get_name(klass));

    S390CPUClass *scc = S390_CPU_CLASS(klass);



    /* strip off the -s390-cpu */

    g_strrstr(name, "-" TYPE_S390_CPU)[0] = 0;

    info = g_new0(CpuDefinitionInfo, 1);

    info->name = name;

    info->has_migration_safe = true;

    info->migration_safe = scc->is_migration_safe;

    info->q_static = scc->is_static;

    info->q_typename = g_strdup(object_class_get_name(klass));

    /* check for unavailable features */

    if (cpu_list_data->model) {

        Object *obj;

        S390CPU *sc;

        obj = object_new(object_class_get_name(klass));

        sc = S390_CPU(obj);

        if (sc->model) {

            info->has_unavailable_features = true;

            check_unavailable_features(cpu_list_data->model, sc->model,

                                       &info->unavailable_features);

        }

        object_unref(obj);

    }



    entry = g_new0(CpuDefinitionInfoList, 1);

    entry->value = info;

    entry->next = *cpu_list;

    *cpu_list = entry;

}
