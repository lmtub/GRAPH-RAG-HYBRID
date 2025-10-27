static void raven_class_init(ObjectClass *klass, void *data)

{

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    k->init = raven_init;

    k->vendor_id = PCI_VENDOR_ID_MOTOROLA;

    k->device_id = PCI_DEVICE_ID_MOTOROLA_RAVEN;

    k->revision = 0x00;

    k->class_id = PCI_CLASS_BRIDGE_HOST;

    dc->desc = "PReP Host Bridge - Motorola Raven";

    dc->vmsd = &vmstate_raven;

    dc->no_user = 1;

}
