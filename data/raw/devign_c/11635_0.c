static int pci_piix_ide_initfn(PCIIDEState *d)

{

    uint8_t *pci_conf = d->dev.config;



    pci_conf[PCI_CLASS_PROG] = 0x80; // legacy ATA mode

    pci_config_set_class(pci_conf, PCI_CLASS_STORAGE_IDE);



    qemu_register_reset(piix3_reset, d);



    pci_register_bar(&d->dev, 4, 0x10, PCI_BASE_ADDRESS_SPACE_IO, bmdma_map);



    vmstate_register(&d->dev.qdev, 0, &vmstate_ide_pci, d);



    pci_piix_init_ports(d);



    return 0;

}
