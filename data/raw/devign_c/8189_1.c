static void xhci_class_init(ObjectClass *klass, void *data)

{

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->vmsd    = &vmstate_xhci;

    dc->props   = xhci_properties;

    dc->reset   = xhci_reset;

    k->init         = usb_xhci_initfn;

    k->vendor_id    = PCI_VENDOR_ID_NEC;

    k->device_id    = PCI_DEVICE_ID_NEC_UPD720200;

    k->class_id     = PCI_CLASS_SERIAL_USB;

    k->revision     = 0x03;

    k->is_express   = 1;


}