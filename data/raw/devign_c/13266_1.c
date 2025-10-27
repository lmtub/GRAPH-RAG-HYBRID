static int kvm_ppc_register_host_cpu_type(void)

{

    TypeInfo type_info = {

        .name = TYPE_HOST_POWERPC_CPU,

        .class_init = kvmppc_host_cpu_class_init,

    };

    PowerPCCPUClass *pvr_pcc;

    DeviceClass *dc;

    int i;



    pvr_pcc = kvm_ppc_get_host_cpu_class();

    if (pvr_pcc == NULL) {

        return -1;

    }

    type_info.parent = object_class_get_name(OBJECT_CLASS(pvr_pcc));

    type_register(&type_info);



#if defined(TARGET_PPC64)

    type_info.name = g_strdup_printf("%s-"TYPE_SPAPR_CPU_CORE, "host");

    type_info.parent = TYPE_SPAPR_CPU_CORE,

    type_info.instance_size = sizeof(sPAPRCPUCore);

    type_info.instance_init = NULL;

    type_info.class_init = spapr_cpu_core_class_init;

    type_info.class_data = (void *) "host";

    type_register(&type_info);

    g_free((void *)type_info.name);

#endif



    /*

     * Update generic CPU family class alias (e.g. on a POWER8NVL host,

     * we want "POWER8" to be a "family" alias that points to the current

     * host CPU type, too)

     */

    dc = DEVICE_CLASS(ppc_cpu_get_family_class(pvr_pcc));

    for (i = 0; ppc_cpu_aliases[i].alias != NULL; i++) {

        if (strcmp(ppc_cpu_aliases[i].alias, dc->desc) == 0) {

            ObjectClass *oc = OBJECT_CLASS(pvr_pcc);

            char *suffix;



            ppc_cpu_aliases[i].model = g_strdup(object_class_get_name(oc));

            suffix = strstr(ppc_cpu_aliases[i].model, "-"TYPE_POWERPC_CPU);

            if (suffix) {

                *suffix = 0;

            }

            ppc_cpu_aliases[i].oc = oc;

            break;

        }

    }



    return 0;

}
