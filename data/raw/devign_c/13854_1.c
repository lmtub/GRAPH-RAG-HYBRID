static void vpb_sic_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = vpb_sic_init;

    dc->no_user = 1;

    dc->vmsd = &vmstate_vpb_sic;

}
