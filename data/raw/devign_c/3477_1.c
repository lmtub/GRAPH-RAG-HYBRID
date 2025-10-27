static void vtd_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    X86IOMMUClass *x86_class = X86_IOMMU_CLASS(klass);



    dc->reset = vtd_reset;

    dc->vmsd = &vtd_vmstate;

    dc->props = vtd_properties;

    dc->hotpluggable = false;

    x86_class->realize = vtd_realize;

    x86_class->int_remap = vtd_int_remap;






}