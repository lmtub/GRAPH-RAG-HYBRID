static int pci_pcnet_init(PCIDevice *pci_dev)

{

    PCIPCNetState *d = PCI_PCNET(pci_dev);

    PCNetState *s = &d->state;

    uint8_t *pci_conf;



#if 0

    printf("sizeof(RMD)=%d, sizeof(TMD)=%d\n",

        sizeof(struct pcnet_RMD), sizeof(struct pcnet_TMD));

#endif



    pci_conf = pci_dev->config;



    pci_set_word(pci_conf + PCI_STATUS,

                 PCI_STATUS_FAST_BACK | PCI_STATUS_DEVSEL_MEDIUM);



    pci_set_word(pci_conf + PCI_SUBSYSTEM_VENDOR_ID, 0x0);

    pci_set_word(pci_conf + PCI_SUBSYSTEM_ID, 0x0);



    pci_conf[PCI_INTERRUPT_PIN] = 1; /* interrupt pin A */

    pci_conf[PCI_MIN_GNT] = 0x06;

    pci_conf[PCI_MAX_LAT] = 0xff;



    /* Handler for memory-mapped I/O */

    memory_region_init_io(&d->state.mmio, OBJECT(d), &pcnet_mmio_ops, s,

                          "pcnet-mmio", PCNET_PNPMMIO_SIZE);



    memory_region_init_io(&d->io_bar, OBJECT(d), &pcnet_io_ops, s, "pcnet-io",

                          PCNET_IOPORT_SIZE);

    pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_IO, &d->io_bar);



    pci_register_bar(pci_dev, 1, 0, &s->mmio);



    s->irq = pci_allocate_irq(pci_dev);

    s->phys_mem_read = pci_physical_memory_read;

    s->phys_mem_write = pci_physical_memory_write;

    s->dma_opaque = pci_dev;



    return pcnet_common_init(DEVICE(pci_dev), s, &net_pci_pcnet_info);

}
