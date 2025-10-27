static void unimp_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = unimp_realize;

    dc->props = unimp_properties;






}