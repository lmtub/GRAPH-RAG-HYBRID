static void acpi_get_hotplug_info(AcpiMiscInfo *misc)

{

    int i;

    PCIBus *bus = find_i440fx();



    if (!bus) {

        /* Only PIIX supports ACPI hotplug */

        memset(misc->slot_hotplug_enable, 0, sizeof misc->slot_hotplug_enable);

        return;

    }



    memset(misc->slot_hotplug_enable, 0xff,

           DIV_ROUND_UP(PCI_SLOT_MAX, BITS_PER_BYTE));



    for (i = 0; i < ARRAY_SIZE(bus->devices); ++i) {

        PCIDeviceClass *pc;

        PCIDevice *pdev = bus->devices[i];



        if (!pdev) {

            continue;

        }



        pc = PCI_DEVICE_GET_CLASS(pdev);



        if (pc->no_hotplug) {

            int slot = PCI_SLOT(i);



            clear_bit(slot, misc->slot_hotplug_enable);

        }

    }

}
