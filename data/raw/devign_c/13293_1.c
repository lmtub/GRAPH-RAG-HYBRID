static void pl190_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = pl190_init;

    dc->no_user = 1;

    dc->reset = pl190_reset;

    dc->vmsd = &vmstate_pl190;

}
