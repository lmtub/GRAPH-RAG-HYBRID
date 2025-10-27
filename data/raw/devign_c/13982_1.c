static void pl181_class_init(ObjectClass *klass, void *data)

{

    SysBusDeviceClass *sdc = SYS_BUS_DEVICE_CLASS(klass);

    DeviceClass *k = DEVICE_CLASS(klass);



    sdc->init = pl181_init;

    k->vmsd = &vmstate_pl181;

    k->reset = pl181_reset;

    k->no_user = 1;

}
