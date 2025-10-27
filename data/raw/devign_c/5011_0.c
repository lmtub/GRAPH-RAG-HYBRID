static void pcibus_dev_print(Monitor *mon, DeviceState *dev, int indent)

{

    PCIDevice *d = (PCIDevice *)dev;

    const pci_class_desc *desc;

    char ctxt[64];

    PCIIORegion *r;

    int i, class;



    class = pci_get_word(d->config + PCI_CLASS_DEVICE);

    desc = pci_class_descriptions;

    while (desc->desc && class != desc->class)

        desc++;

    if (desc->desc) {

        snprintf(ctxt, sizeof(ctxt), "%s", desc->desc);

    } else {

        snprintf(ctxt, sizeof(ctxt), "Class %04x", class);

    }



    monitor_printf(mon, "%*sclass %s, addr %02x:%02x.%x, "

                   "pci id %04x:%04x (sub %04x:%04x)\n",

                   indent, "", ctxt, pci_bus_num(d->bus),

                   PCI_SLOT(d->devfn), PCI_FUNC(d->devfn),

                   pci_get_word(d->config + PCI_VENDOR_ID),

                   pci_get_word(d->config + PCI_DEVICE_ID),

                   pci_get_word(d->config + PCI_SUBSYSTEM_VENDOR_ID),

                   pci_get_word(d->config + PCI_SUBSYSTEM_ID));

    for (i = 0; i < PCI_NUM_REGIONS; i++) {

        r = &d->io_regions[i];

        if (!r->size)

            continue;

        monitor_printf(mon, "%*sbar %d: %s at 0x%"FMT_PCIBUS

                       " [0x%"FMT_PCIBUS"]\n",

                       indent, "",

                       i, r->type & PCI_BASE_ADDRESS_SPACE_IO ? "i/o" : "mem",

                       r->addr, r->addr + r->size - 1);

    }

}
