static int vt82c686b_ide_initfn(PCIDevice *dev)

{

    PCIIDEState *d = DO_UPCAST(PCIIDEState, dev, dev);;

    uint8_t *pci_conf = d->dev.config;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_VIA);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_VIA_IDE);

    pci_config_set_class(pci_conf, PCI_CLASS_STORAGE_IDE);

    pci_config_set_prog_interface(pci_conf, 0x8a); /* legacy ATA mode */

    pci_config_set_revision(pci_conf,0x06); /* Revision 0.6 */

    pci_set_long(pci_conf + PCI_CAPABILITY_LIST, 0x000000c0);



    qemu_register_reset(via_reset, d);

    pci_register_bar(&d->dev, 4, 0x10,

                           PCI_BASE_ADDRESS_SPACE_IO, bmdma_map);



    vmstate_register(&dev->qdev, 0, &vmstate_ide_pci, d);



    vt82c686b_init_ports(d);



    return 0;

}
