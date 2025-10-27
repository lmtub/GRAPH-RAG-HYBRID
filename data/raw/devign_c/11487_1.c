static void port92_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->no_user = 1;

    dc->realize = port92_realizefn;

    dc->reset = port92_reset;

    dc->vmsd = &vmstate_port92_isa;

}
