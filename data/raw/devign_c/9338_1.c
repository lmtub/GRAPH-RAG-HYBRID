static void pit_common_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = pit_common_realize;

    dc->vmsd = &vmstate_pit_common;

    dc->no_user = 1;

}
