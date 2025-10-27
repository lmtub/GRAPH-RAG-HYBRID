static void vmport_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = vmport_realizefn;

    dc->no_user = 1;

}
