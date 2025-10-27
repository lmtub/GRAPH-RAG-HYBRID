static void acpi_pcihp_eject_slot(AcpiPciHpState *s, unsigned bsel, unsigned slots)

{

    BusChild *kid, *next;

    int slot = ffs(slots) - 1;

    PCIBus *bus = acpi_pcihp_find_hotplug_bus(s, bsel);



    if (!bus) {

        return;

    }



    /* Mark request as complete */

    s->acpi_pcihp_pci_status[bsel].down &= ~(1U << slot);

    s->acpi_pcihp_pci_status[bsel].up &= ~(1U << slot);



    QTAILQ_FOREACH_SAFE(kid, &bus->qbus.children, sibling, next) {

        DeviceState *qdev = kid->child;

        PCIDevice *dev = PCI_DEVICE(qdev);

        if (PCI_SLOT(dev->devfn) == slot) {

            if (!acpi_pcihp_pc_no_hotplug(s, dev)) {

                object_unparent(OBJECT(qdev));

            }

        }

    }

}
