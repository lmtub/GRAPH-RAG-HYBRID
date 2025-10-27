static void via_ide_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = vt82c686b_ide_initfn;

    k->exit = vt82c686b_ide_exitfn;

    k->vendor_id = PCI_VENDOR_ID_VIA;

    k->device_id = PCI_DEVICE_ID_VIA_IDE;

    k->revision = 0x06;

    k->class_id = PCI_CLASS_STORAGE_IDE;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

    dc->no_user = 1;

}
