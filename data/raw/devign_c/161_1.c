static void xendev_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->props = xendev_properties;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);






}