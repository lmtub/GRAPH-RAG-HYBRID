static void bonito_pcihost_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = bonito_pcihost_initfn;

    dc->no_user = 1;

}
