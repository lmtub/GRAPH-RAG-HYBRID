static void xics_common_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);




    dc->reset = xics_common_reset;


}