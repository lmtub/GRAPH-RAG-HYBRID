static void sun4m_fdc_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->props = sun4m_fdc_properties;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);






}