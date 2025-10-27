static void hpet_device_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = hpet_realize;

    dc->reset = hpet_reset;

    dc->vmsd = &vmstate_hpet;

    dc->props = hpet_device_properties;






}