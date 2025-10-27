PCIBus *pci_prep_init(qemu_irq *pic)

{

    PREPPCIState *s;

    PCIDevice *d;

    int PPC_io_memory;



    s = qemu_mallocz(sizeof(PREPPCIState));

    s->bus = pci_register_bus(prep_set_irq, prep_map_irq, pic, 0, 2);



    register_ioport_write(0xcf8, 4, 4, pci_prep_addr_writel, s);

    register_ioport_read(0xcf8, 4, 4, pci_prep_addr_readl, s);



    register_ioport_write(0xcfc, 4, 1, pci_host_data_writeb, s);

    register_ioport_write(0xcfc, 4, 2, pci_host_data_writew, s);

    register_ioport_write(0xcfc, 4, 4, pci_host_data_writel, s);

    register_ioport_read(0xcfc, 4, 1, pci_host_data_readb, s);

    register_ioport_read(0xcfc, 4, 2, pci_host_data_readw, s);

    register_ioport_read(0xcfc, 4, 4, pci_host_data_readl, s);



    PPC_io_memory = cpu_register_io_memory(0, PPC_PCIIO_read,

                                           PPC_PCIIO_write, s);

    cpu_register_physical_memory(0x80800000, 0x00400000, PPC_io_memory);



    /* PCI host bridge */

    d = pci_register_device(s->bus, "PREP Host Bridge - Motorola Raven",

                            sizeof(PCIDevice), 0, NULL, NULL);

    d->config[0x00] = 0x57; // vendor_id : Motorola

    d->config[0x01] = 0x10;

    d->config[0x02] = 0x01; // device_id : Raven

    d->config[0x03] = 0x48;

    d->config[0x08] = 0x00; // revision

    d->config[0x0A] = 0x00; // class_sub = pci host

    d->config[0x0B] = 0x06; // class_base = PCI_bridge

    d->config[0x0C] = 0x08; // cache_line_size

    d->config[0x0D] = 0x10; // latency_timer

    d->config[0x0E] = 0x00; // header_type

    d->config[0x34] = 0x00; // capabilities_pointer



    return s->bus;

}
