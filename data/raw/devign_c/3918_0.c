static int kvm_ppc_register_host_cpu_type(void)

{

    TypeInfo type_info = {

        .name = TYPE_HOST_POWERPC_CPU,

        .instance_init = kvmppc_host_cpu_initfn,

        .class_init = kvmppc_host_cpu_class_init,

    };

    PowerPCCPUClass *pvr_pcc;

    DeviceClass *dc;



    pvr_pcc = kvm_ppc_get_host_cpu_class();

    if (pvr_pcc == NULL) {

        return -1;

    }

    type_info.parent = object_class_get_name(OBJECT_CLASS(pvr_pcc));

    type_register(&type_info);



    /* Register generic family CPU class for a family */

    pvr_pcc = ppc_cpu_get_family_class(pvr_pcc);

    dc = DEVICE_CLASS(pvr_pcc);

    type_info.parent = object_class_get_name(OBJECT_CLASS(pvr_pcc));

    type_info.name = g_strdup_printf("%s-"TYPE_POWERPC_CPU, dc->desc);

    type_register(&type_info);



#if defined(TARGET_PPC64)

    type_info.name = g_strdup_printf("%s-"TYPE_SPAPR_CPU_CORE, "host");

    type_info.parent = TYPE_SPAPR_CPU_CORE,

    type_info.instance_size = sizeof(sPAPRCPUCore),

    type_info.instance_init = spapr_cpu_core_host_initfn,

    type_info.class_init = NULL;

    type_register(&type_info);

    g_free((void *)type_info.name);



    /* Register generic spapr CPU family class for current host CPU type */

    type_info.name = g_strdup_printf("%s-"TYPE_SPAPR_CPU_CORE, dc->desc);

    type_register(&type_info);

    g_free((void *)type_info.name);

#endif



    return 0;

}
