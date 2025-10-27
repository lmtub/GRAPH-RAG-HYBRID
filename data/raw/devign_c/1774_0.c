static int nic_init(PCIDevice *pci_dev, uint32_t device)

{

    PCIEEPRO100State *d = DO_UPCAST(PCIEEPRO100State, dev, pci_dev);

    EEPRO100State *s;



    logout("\n");



    d->dev.unregister = pci_nic_uninit;



    s = &d->eepro100;

    s->device = device;

    s->pci_dev = &d->dev;



    pci_reset(s);



    /* Add 64 * 2 EEPROM. i82557 and i82558 support a 64 word EEPROM,

     * i82559 and later support 64 or 256 word EEPROM. */

    s->eeprom = eeprom93xx_new(EEPROM_SIZE);



    /* Handler for memory-mapped I/O */

    d->eepro100.mmio_index =

        cpu_register_io_memory(pci_mmio_read, pci_mmio_write, s);



    pci_register_bar(&d->dev, 0, PCI_MEM_SIZE,

                           PCI_ADDRESS_SPACE_MEM |

                           PCI_ADDRESS_SPACE_MEM_PREFETCH, pci_mmio_map);

    pci_register_bar(&d->dev, 1, PCI_IO_SIZE, PCI_ADDRESS_SPACE_IO,

                           pci_map);

    pci_register_bar(&d->dev, 2, PCI_FLASH_SIZE, PCI_ADDRESS_SPACE_MEM,

                           pci_mmio_map);



    qdev_get_macaddr(&d->dev.qdev, s->macaddr);

    logout("macaddr: %s\n", nic_dump(&s->macaddr[0], 6));

    assert(s->region[1] == 0);



    nic_reset(s);



    s->vc = qdev_get_vlan_client(&d->dev.qdev,

                                 nic_can_receive, nic_receive, NULL,

                                 nic_cleanup, s);



    qemu_format_nic_info_str(s->vc, s->macaddr);



    qemu_register_reset(nic_reset, s);



    register_savevm(s->vc->model, -1, 3, nic_save, nic_load, s);

    return 0;

}
