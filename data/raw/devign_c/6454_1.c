static void pl110_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = pl110_initfn;

    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);

    dc->no_user = 1;

    dc->vmsd = &vmstate_pl110;

}
