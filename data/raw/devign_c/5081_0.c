static int grackle_pci_host_init(PCIDevice *d)

{

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_MOTOROLA);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_MOTOROLA_MPC106);

    d->config[0x08] = 0x00; // revision

    d->config[0x09] = 0x01;

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_HOST);

    return 0;

}
