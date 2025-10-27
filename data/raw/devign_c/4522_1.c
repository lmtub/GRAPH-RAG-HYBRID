static void etsec_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = etsec_realize;

    dc->reset = etsec_reset;

    dc->props = etsec_properties;



}