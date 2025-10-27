static void apic_common_class_init(ObjectClass *klass, void *data)

{

    ICCDeviceClass *idc = ICC_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->vmsd = &vmstate_apic_common;

    dc->reset = apic_reset_common;

    dc->no_user = 1;

    dc->props = apic_properties_common;

    idc->init = apic_init_common;

}
