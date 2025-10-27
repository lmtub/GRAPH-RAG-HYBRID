static void xen_sysdev_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SysBusDeviceClass *k = SYS_BUS_DEVICE_CLASS(klass);



    k->init = xen_sysdev_init;

    dc->props = xen_sysdev_properties;

    dc->bus_type = TYPE_XENSYSBUS;






}