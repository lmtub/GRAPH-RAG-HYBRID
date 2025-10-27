static void arm_cpu_register_types(void)

{

    int i;



    type_register_static(&arm_cpu_type_info);

    for (i = 0; i < ARRAY_SIZE(arm_cpus); i++) {

        cpu_register(&arm_cpus[i]);

    }

}
