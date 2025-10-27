static void aarch64_cpu_register(const ARMCPUInfo *info)

{

    TypeInfo type_info = {

        .parent = TYPE_AARCH64_CPU,

        .instance_size = sizeof(ARMCPU),

        .instance_init = info->initfn,

        .class_size = sizeof(ARMCPUClass),

        .class_init = info->class_init,

    };



    /* TODO: drop when we support more CPUs - all entries will have name set */

    if (!info->name) {

        return;

    }



    type_info.name = g_strdup_printf("%s-" TYPE_ARM_CPU, info->name);

    type_register(&type_info);

    g_free((void *)type_info.name);

}
