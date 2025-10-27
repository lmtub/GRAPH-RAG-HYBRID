static int dec_21154_pci_host_init(PCIDevice *d)

{

    /* PCI2PCI bridge same values as PearPC - check this */

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_DEC);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_DEC_21154);

    pci_set_byte(d->config + PCI_REVISION_ID, 0x02);

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_PCI);

    return 0;

}
