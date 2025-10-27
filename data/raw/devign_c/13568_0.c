static int gt64120_pci_init(PCIDevice *d)

{

    /* FIXME: Malta specific hw assumptions ahead */

    pci_config_set_vendor_id(d->config, PCI_VENDOR_ID_MARVELL);

    pci_config_set_device_id(d->config, PCI_DEVICE_ID_MARVELL_GT6412X);

    pci_set_word(d->config + PCI_COMMAND, 0);

    pci_set_word(d->config + PCI_STATUS,

                 PCI_STATUS_FAST_BACK | PCI_STATUS_DEVSEL_MEDIUM);

    pci_set_byte(d->config + PCI_CLASS_REVISION, 0x10);

    pci_config_set_prog_interface(d->config, 0);

    pci_config_set_class(d->config, PCI_CLASS_BRIDGE_HOST);

    pci_set_long(d->config + PCI_BASE_ADDRESS_0, 0x00000008);

    pci_set_long(d->config + PCI_BASE_ADDRESS_1, 0x01000008);

    pci_set_long(d->config + PCI_BASE_ADDRESS_2, 0x1c000000);

    pci_set_long(d->config + PCI_BASE_ADDRESS_3, 0x1f000000);

    pci_set_long(d->config + PCI_BASE_ADDRESS_4, 0x14000000);

    pci_set_long(d->config + PCI_BASE_ADDRESS_5, 0x14000001);

    pci_set_byte(d->config + 0x3d, 0x01);



    return 0;

}
