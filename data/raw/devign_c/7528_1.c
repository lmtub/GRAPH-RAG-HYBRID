static void pl041_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = pl041_init;

    set_bit(DEVICE_CATEGORY_SOUND, dc->categories);

    dc->no_user = 1;

    dc->reset = pl041_device_reset;

    dc->vmsd = &vmstate_pl041;

    dc->props = pl041_device_properties;

}
