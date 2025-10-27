static void sysbus_fdc_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->props = sysbus_fdc_properties;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);






}