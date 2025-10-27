static void pci_grackle_class_init(ObjectClass *klass, void *data)

{

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    k->init = pci_grackle_init_device;

    dc->no_user = 1;

}
