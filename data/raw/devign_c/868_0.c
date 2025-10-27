static void xics_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);



    dc->realize = xics_realize;

    dc->props = xics_properties;

    dc->reset = xics_reset;

}
