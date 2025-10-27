static void ide_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *k = DEVICE_CLASS(klass);

    k->init = ide_qdev_init;

    set_bit(DEVICE_CATEGORY_STORAGE, k->categories);

    k->bus_type = TYPE_IDE_BUS;


    k->props = ide_props;

}