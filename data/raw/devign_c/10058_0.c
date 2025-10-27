static void spapr_cpu_core_register(const SPAPRCoreInfo *info)

{

    TypeInfo type_info = {

        .parent = TYPE_SPAPR_CPU_CORE,

        .instance_size = sizeof(sPAPRCPUCore),

        .instance_init = info->initfn,

    };



    type_info.name = g_strdup_printf("%s-" TYPE_SPAPR_CPU_CORE, info->name);

    type_register(&type_info);

    g_free((void *)type_info.name);

}
