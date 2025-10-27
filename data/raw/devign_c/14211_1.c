static void kvm_s390_flic_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    S390FLICStateClass *fsc = S390_FLIC_COMMON_CLASS(oc);




    dc->realize = kvm_s390_flic_realize;

    dc->vmsd = &kvm_s390_flic_vmstate;

    dc->reset = kvm_s390_flic_reset;

    fsc->register_io_adapter = kvm_s390_register_io_adapter;

    fsc->io_adapter_map = kvm_s390_io_adapter_map;

    fsc->add_adapter_routes = kvm_s390_add_adapter_routes;

    fsc->release_adapter_routes = kvm_s390_release_adapter_routes;

    fsc->clear_io_irq = kvm_s390_clear_io_flic;

}