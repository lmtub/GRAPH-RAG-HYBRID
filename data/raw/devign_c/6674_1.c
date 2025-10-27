static void xen_pci_passthrough_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    k->realize = xen_pt_realize;

    k->exit = xen_pt_unregister_device;

    k->config_read = xen_pt_pci_read_config;

    k->config_write = xen_pt_pci_write_config;


    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    dc->desc = "Assign an host PCI device with Xen";

    dc->props = xen_pci_passthrough_properties;

};