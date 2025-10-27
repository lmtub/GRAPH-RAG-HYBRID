static void spapr_cpu_core_register_types(void)

{

    const SPAPRCoreInfo *info = spapr_cores;



    type_register_static(&spapr_cpu_core_type_info);

    while (info->name) {

        spapr_cpu_core_register(info);

        info++;

    }

}
