int acpi_add_cpu_info(Object *o, void *opaque)

{

    AcpiCpuInfo *cpu = opaque;

    uint64_t apic_id;



    if (object_dynamic_cast(o, TYPE_CPU)) {

        apic_id = object_property_get_int(o, "apic-id", NULL);

        assert(apic_id <= MAX_CPUMASK_BITS);



        set_bit(apic_id, cpu->found_cpus);

    }



    object_child_foreach(o, acpi_add_cpu_info, opaque);

    return 0;

}
