static void acpi_set_pci_info(void)

{

    PCIBus *bus = find_i440fx(); /* TODO: Q35 support */

    unsigned bsel_alloc = 0;



    if (bus) {

        /* Scan all PCI buses. Set property to enable acpi based hotplug. */

        pci_for_each_bus_depth_first(bus, acpi_set_bsel, NULL, &bsel_alloc);

    }

}
