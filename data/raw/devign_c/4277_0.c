static void pci_vpb_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = pci_vpb_realize;

    dc->reset = pci_vpb_reset;

    dc->vmsd = &pci_vpb_vmstate;

    dc->props = pci_vpb_properties;

    /* Reason: object_unref() hangs */

    dc->cannot_destroy_with_object_finalize_yet = true;

}
