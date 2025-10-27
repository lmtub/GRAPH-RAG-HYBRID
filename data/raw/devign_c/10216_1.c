static void amdvi_class_init(ObjectClass *klass, void* data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    X86IOMMUClass *dc_class = X86_IOMMU_CLASS(klass);



    dc->reset = amdvi_reset;

    dc->vmsd = &vmstate_amdvi;

    dc->hotpluggable = false;

    dc_class->realize = amdvi_realize;






}