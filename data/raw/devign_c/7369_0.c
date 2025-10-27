void pc_cpus_init(const char *cpu_model, DeviceState *icc_bridge)

{

    int i;

    X86CPU *cpu = NULL;

    Error *error = NULL;

    unsigned long apic_id_limit;



    /* init CPUs */

    if (cpu_model == NULL) {

#ifdef TARGET_X86_64

        cpu_model = "qemu64";

#else

        cpu_model = "qemu32";

#endif

    }

    current_cpu_model = cpu_model;



    apic_id_limit = pc_apic_id_limit(max_cpus);

    if (apic_id_limit > ACPI_CPU_HOTPLUG_ID_LIMIT) {

        error_report("max_cpus is too large. APIC ID of last CPU is %lu",

                     apic_id_limit - 1);

        exit(1);

    }



    for (i = 0; i < smp_cpus; i++) {

        cpu = pc_new_cpu(cpu_model, x86_cpu_apic_id_from_index(i),

                         icc_bridge, &error);

        if (error) {

            error_report_err(error);

            exit(1);

        }

        object_unref(OBJECT(cpu));

    }



    /* tell smbios about cpuid version and features */

    smbios_set_cpuid(cpu->env.cpuid_version, cpu->env.features[FEAT_1_EDX]);

}
