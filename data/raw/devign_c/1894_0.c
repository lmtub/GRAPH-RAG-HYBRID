static int xen_platform_initfn(PCIDevice *dev)

{

    PCIXenPlatformState *d = DO_UPCAST(PCIXenPlatformState, pci_dev, dev);

    uint8_t *pci_conf;



    pci_conf = d->pci_dev.config;



    pci_set_word(pci_conf + PCI_COMMAND, PCI_COMMAND_IO | PCI_COMMAND_MEMORY);



    pci_config_set_prog_interface(pci_conf, 0);



    pci_conf[PCI_INTERRUPT_PIN] = 1;



    pci_register_bar(&d->pci_dev, 0, 0x100,

            PCI_BASE_ADDRESS_SPACE_IO, platform_ioport_map);



    /* reserve 16MB mmio address for share memory*/

    pci_register_bar(&d->pci_dev, 1, 0x1000000,

            PCI_BASE_ADDRESS_MEM_PREFETCH, platform_mmio_map);



    platform_fixed_ioport_init(d);



    return 0;

}
