static void spapr_cpu_core_host_initfn(Object *obj)

{

    sPAPRCPUCore *core = SPAPR_CPU_CORE(obj);

    char *name = g_strdup_printf("%s-" TYPE_POWERPC_CPU, "host");

    ObjectClass *oc = object_class_by_name(name);



    g_assert(oc);

    g_free((void *)name);

    core->cpu_class = oc;

}
