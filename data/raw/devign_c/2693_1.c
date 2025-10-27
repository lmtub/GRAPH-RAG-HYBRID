static void piix3_ide_xen_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = pci_piix_ide_initfn;

    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_82371SB_1;

    k->class_id = PCI_CLASS_STORAGE_IDE;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

    dc->no_user = 1;

    dc->unplug = pci_piix3_xen_ide_unplug;

}
