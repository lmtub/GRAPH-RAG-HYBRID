void acpi_memory_hotplug_init(MemoryRegion *as, Object *owner,

                              MemHotplugState *state)

{

    MachineState *machine = MACHINE(qdev_get_machine());



    state->dev_count = machine->ram_slots;

    if (!state->dev_count) {

        return;

    }



    state->devs = g_malloc0(sizeof(*state->devs) * state->dev_count);

    memory_region_init_io(&state->io, owner, &acpi_memory_hotplug_ops, state,

                          "acpi-mem-hotplug", ACPI_MEMORY_HOTPLUG_IO_LEN);

    memory_region_add_subregion(as, ACPI_MEMORY_HOTPLUG_BASE, &state->io);

}
