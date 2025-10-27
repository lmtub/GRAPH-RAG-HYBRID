void pc_cpus_init(PCMachineState *pcms)

{

    int i;

    CPUClass *cc;

    ObjectClass *oc;

    const char *typename;

    gchar **model_pieces;

    X86CPU *cpu = NULL;

    MachineState *machine = MACHINE(pcms);



    /* init CPUs */

    if (machine->cpu_model == NULL) {

#ifdef TARGET_X86_64

        machine->cpu_model = "qemu64";

#else

        machine->cpu_model = "qemu32";

#endif

    }



    model_pieces = g_strsplit(machine->cpu_model, ",", 2);

    if (!model_pieces[0]) {

        error_report("Invalid/empty CPU model name");

        exit(1);

    }



    oc = cpu_class_by_name(TYPE_X86_CPU, model_pieces[0]);

    if (oc == NULL) {

        error_report("Unable to find CPU definition: %s", model_pieces[0]);

        exit(1);

    }

    typename = object_class_get_name(oc);

    cc = CPU_CLASS(oc);

    cc->parse_features(typename, model_pieces[1], &error_fatal);

    g_strfreev(model_pieces);



    /* Calculates the limit to CPU APIC ID values

     *

     * Limit for the APIC ID value, so that all

     * CPU APIC IDs are < pcms->apic_id_limit.

     *

     * This is used for FW_CFG_MAX_CPUS. See comments on bochs_bios_init().

     */

    pcms->apic_id_limit = x86_cpu_apic_id_from_index(max_cpus - 1) + 1;

    if (pcms->apic_id_limit > ACPI_CPU_HOTPLUG_ID_LIMIT) {

        error_report("max_cpus is too large. APIC ID of last CPU is %u",

                     pcms->apic_id_limit - 1);

        exit(1);

    }



    pcms->possible_cpus = g_malloc0(sizeof(CPUArchIdList) +

                                    sizeof(CPUArchId) * max_cpus);

    for (i = 0; i < max_cpus; i++) {

        pcms->possible_cpus->cpus[i].arch_id = x86_cpu_apic_id_from_index(i);

        pcms->possible_cpus->len++;

        if (i < smp_cpus) {

            cpu = pc_new_cpu(typename, x86_cpu_apic_id_from_index(i),

                             &error_fatal);

            object_unref(OBJECT(cpu));

        }

    }



    /* tell smbios about cpuid version and features */

    smbios_set_cpuid(cpu->env.cpuid_version, cpu->env.features[FEAT_1_EDX]);

}
