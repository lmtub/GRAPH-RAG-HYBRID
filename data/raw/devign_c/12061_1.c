static void via_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = vt82c686b_initfn;

    k->config_write = vt82c686b_write_config;

    k->vendor_id = PCI_VENDOR_ID_VIA;

    k->device_id = PCI_DEVICE_ID_VIA_ISA_BRIDGE;

    k->class_id = PCI_CLASS_BRIDGE_ISA;

    k->revision = 0x40;

    dc->desc = "ISA bridge";

    dc->no_user = 1;

    dc->vmsd = &vmstate_via;

}
