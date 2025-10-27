static void s390_print_cpu_model_list_entry(gpointer data, gpointer user_data)

{

    CPUListState *s = user_data;

    const S390CPUClass *scc = S390_CPU_CLASS((ObjectClass *)data);

    char *name = g_strdup(object_class_get_name((ObjectClass *)data));

    const char *details = "";



    if (scc->is_static) {

        details = "(static, migration-safe)";

    } else if (scc->is_migration_safe) {

        details = "(migration-safe)";

    }



    /* strip off the -s390-cpu */

    g_strrstr(name, "-" TYPE_S390_CPU)[0] = 0;

    (*s->cpu_fprintf)(s->file, "s390 %-15s %-35s %s\n", name, scc->desc,

                      details);

    g_free(name);

}
