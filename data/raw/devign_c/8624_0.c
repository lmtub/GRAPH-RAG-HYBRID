static int unin_agp_pci_host_init(PCIDevice *d)

{

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_APPLE);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_APPLE_UNI_N_AGP);

    d->config[0x08] = 0x00; // revision

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_HOST);

    d->config[0x0C] = 0x08; // cache_line_size

    d->config[0x0D] = 0x10; // latency_timer

    //    d->config[0x34] = 0x80; // capabilities_pointer

    return 0;

}
