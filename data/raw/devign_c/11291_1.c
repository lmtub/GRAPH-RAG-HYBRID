static void vmmouse_class_initfn(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = vmmouse_realizefn;

    dc->no_user = 1;

    dc->reset = vmmouse_reset;

    dc->vmsd = &vmstate_vmmouse;

    dc->props = vmmouse_properties;

}
