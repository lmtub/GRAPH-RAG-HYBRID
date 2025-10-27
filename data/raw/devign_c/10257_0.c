static int pci_pbm_init_device(SysBusDevice *dev)

{



    APBState *s;

    int pci_mem_config, pci_mem_data, apb_config, pci_ioport;



    s = FROM_SYSBUS(APBState, dev);

    /* apb_config */

    apb_config = cpu_register_io_memory(apb_config_read,

                                        apb_config_write, s);

    sysbus_init_mmio(dev, 0x40ULL, apb_config);

    /* pci_ioport */

    pci_ioport = cpu_register_io_memory(pci_apb_ioread,

                                          pci_apb_iowrite, s);

    sysbus_init_mmio(dev, 0x10000ULL, pci_ioport);

    /* mem_config  */

    pci_mem_config = cpu_register_io_memory(pci_apb_config_read,

                                            pci_apb_config_write, s);

    sysbus_init_mmio(dev, 0x10ULL, pci_mem_config);

    /* mem_data */

    pci_mem_data = cpu_register_io_memory(pci_apb_read,

                                          pci_apb_write, &s->host_state);

    sysbus_init_mmio(dev, 0x10000000ULL, pci_mem_data);

    return 0;

}
