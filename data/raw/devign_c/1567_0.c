static void acpi_pcihp_update_hotplug_bus(AcpiPciHpState *s, int bsel)

{

    BusChild *kid, *next;

    PCIBus *bus = acpi_pcihp_find_hotplug_bus(s, bsel);



    /* Execute any pending removes during reset */

    while (s->acpi_pcihp_pci_status[bsel].down) {

        acpi_pcihp_eject_slot(s, bsel, s->acpi_pcihp_pci_status[bsel].down);

    }



    s->acpi_pcihp_pci_status[bsel].hotplug_enable = ~0;

    s->acpi_pcihp_pci_status[bsel].device_present = 0;



    if (!bus) {

        return;

    }

    QTAILQ_FOREACH_SAFE(kid, &bus->qbus.children, sibling, next) {

        DeviceState *qdev = kid->child;

        PCIDevice *pdev = PCI_DEVICE(qdev);

        int slot = PCI_SLOT(pdev->devfn);



        if (acpi_pcihp_pc_no_hotplug(s, pdev)) {

            s->acpi_pcihp_pci_status[bsel].hotplug_enable &= ~(1U << slot);

        }



        s->acpi_pcihp_pci_status[bsel].device_present |= (1U << slot);

    }

}
