void pc_hot_add_cpu(const int64_t id, Error **errp)

{

    X86CPU *cpu;

    ObjectClass *oc;

    PCMachineState *pcms = PC_MACHINE(qdev_get_machine());

    int64_t apic_id = x86_cpu_apic_id_from_index(id);

    Error *local_err = NULL;



    if (id < 0) {

        error_setg(errp, "Invalid CPU id: %" PRIi64, id);

        return;

    }



    if (cpu_exists(apic_id)) {

        error_setg(errp, "Unable to add CPU: %" PRIi64

                   ", it already exists", id);

        return;

    }



    if (id >= max_cpus) {

        error_setg(errp, "Unable to add CPU: %" PRIi64

                   ", max allowed: %d", id, max_cpus - 1);

        return;

    }



    if (apic_id >= ACPI_CPU_HOTPLUG_ID_LIMIT) {

        error_setg(errp, "Unable to add CPU: %" PRIi64

                   ", resulting APIC ID (%" PRIi64 ") is too large",

                   id, apic_id);

        return;

    }



    assert(pcms->possible_cpus->cpus[0].cpu); /* BSP is always present */

    oc = OBJECT_CLASS(CPU_GET_CLASS(pcms->possible_cpus->cpus[0].cpu));

    cpu = pc_new_cpu(object_class_get_name(oc), apic_id, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }

    object_unref(OBJECT(cpu));

}
