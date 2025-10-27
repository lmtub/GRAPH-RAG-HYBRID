static void s390_ipl_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = s390_ipl_init;

    dc->props = s390_ipl_properties;

    dc->reset = s390_ipl_reset;

    dc->no_user = 1;

}
