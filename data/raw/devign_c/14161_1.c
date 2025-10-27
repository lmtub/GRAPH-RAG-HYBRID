static void typhoon_pcihost_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = typhoon_pcihost_init;

    dc->no_user = 1;

}
