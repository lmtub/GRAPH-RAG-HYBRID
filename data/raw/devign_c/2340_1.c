static void ioapic_common_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = ioapic_common_realize;

    dc->vmsd = &vmstate_ioapic_common;

    dc->no_user = 1;

}
