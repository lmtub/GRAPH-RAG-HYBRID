int acpi_pcihp_device_hotplug(AcpiPciHpState *s, PCIDevice *dev,

                              PCIHotplugState state)

{

    int slot = PCI_SLOT(dev->devfn);

    int bsel = acpi_pcihp_get_bsel(dev->bus);

    if (bsel < 0) {

        return -1;

    }



    /* Don't send event when device is enabled during qemu machine creation:

     * it is present on boot, no hotplug event is necessary. We do send an

     * event when the device is disabled later. */

    if (state == PCI_COLDPLUG_ENABLED) {

        s->acpi_pcihp_pci_status[bsel].device_present |= (1U << slot);

        return 0;

    }



    if (state == PCI_HOTPLUG_ENABLED) {

        enable_device(s, bsel, slot);

    } else {

        disable_device(s, bsel, slot);

    }



    return 0;

}
