static void ohci_pci_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->init = usb_ohci_initfn_pci;

    k->vendor_id = PCI_VENDOR_ID_APPLE;

    k->device_id = PCI_DEVICE_ID_APPLE_IPID_USB;

    k->class_id = PCI_CLASS_SERIAL_USB;


    dc->desc = "Apple USB Controller";

    dc->props = ohci_pci_properties;

}