void acpi_pcihp_init(Object *owner, AcpiPciHpState *s, PCIBus *root_bus,

                     MemoryRegion *address_space_io, bool bridges_enabled)

{

    s->io_len = ACPI_PCIHP_SIZE;

    s->io_base = ACPI_PCIHP_ADDR;



    s->root= root_bus;

    s->legacy_piix = !bridges_enabled;



    if (s->legacy_piix) {

        unsigned *bus_bsel = g_malloc(sizeof *bus_bsel);



        s->io_len = ACPI_PCIHP_LEGACY_SIZE;



        *bus_bsel = ACPI_PCIHP_BSEL_DEFAULT;

        object_property_add_uint32_ptr(OBJECT(root_bus), ACPI_PCIHP_PROP_BSEL,

                                       bus_bsel, NULL);

    }



    memory_region_init_io(&s->io, owner, &acpi_pcihp_io_ops, s,

                          "acpi-pci-hotplug", s->io_len);

    memory_region_add_subregion(address_space_io, s->io_base, &s->io);



    object_property_add_uint16_ptr(owner, ACPI_PCIHP_IO_BASE_PROP, &s->io_base,

                                   &error_abort);

    object_property_add_uint16_ptr(owner, ACPI_PCIHP_IO_LEN_PROP, &s->io_len,

                                   &error_abort);

}
