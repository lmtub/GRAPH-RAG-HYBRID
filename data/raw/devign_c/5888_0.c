static void unplug_disks(PCIBus *b, PCIDevice *d, void *o)

{

    /* We have to ignore passthrough devices */

    if (!strcmp(d->name, "xen-pci-passthrough")) {

        return;

    }



    switch (pci_get_word(d->config + PCI_CLASS_DEVICE)) {

    case PCI_CLASS_STORAGE_IDE:

        pci_piix3_xen_ide_unplug(DEVICE(d));

        break;



    case PCI_CLASS_STORAGE_SCSI:

    case PCI_CLASS_STORAGE_EXPRESS:

        object_unparent(OBJECT(d));

        break;



    default:

        break;

    }

}
