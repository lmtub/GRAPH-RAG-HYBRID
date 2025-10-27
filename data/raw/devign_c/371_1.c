static void arm_gic_common_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->reset = arm_gic_common_reset;

    dc->realize = arm_gic_common_realize;

    dc->props = arm_gic_common_properties;

    dc->vmsd = &vmstate_gic;

    dc->no_user = 1;

}
