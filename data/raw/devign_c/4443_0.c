static int pci_rtl8139_init(PCIDevice *dev)

{

    RTL8139State * s = DO_UPCAST(RTL8139State, dev, dev);

    uint8_t *pci_conf;



    pci_conf = s->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_REALTEK);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_REALTEK_8139);

    pci_conf[PCI_REVISION_ID] = RTL8139_PCI_REVID; /* >=0x20 is for 8139C+ */

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    pci_conf[PCI_INTERRUPT_PIN] = 1;    /* interrupt pin 0 */

    /* TODO: start of capability list, but no capability

     * list bit in status register, and offset 0xdc seems unused. */

    pci_conf[PCI_CAPABILITY_LIST] = 0xdc;



    /* I/O handler for memory-mapped I/O */

    s->rtl8139_mmio_io_addr =

        cpu_register_io_memory(rtl8139_mmio_read, rtl8139_mmio_write, s,

                               DEVICE_LITTLE_ENDIAN);



    pci_register_bar(&s->dev, 0, 0x100,

                           PCI_BASE_ADDRESS_SPACE_IO,  rtl8139_ioport_map);



    pci_register_bar_simple(&s->dev, 1, 0x100, 0, s->rtl8139_mmio_io_addr);



    qemu_macaddr_default_if_unset(&s->conf.macaddr);



    /* prepare eeprom */

    s->eeprom.contents[0] = 0x8129;

#if 1

    /* PCI vendor and device ID should be mirrored here */

    s->eeprom.contents[1] = PCI_VENDOR_ID_REALTEK;

    s->eeprom.contents[2] = PCI_DEVICE_ID_REALTEK_8139;

#endif

    s->eeprom.contents[7] = s->conf.macaddr.a[0] | s->conf.macaddr.a[1] << 8;

    s->eeprom.contents[8] = s->conf.macaddr.a[2] | s->conf.macaddr.a[3] << 8;

    s->eeprom.contents[9] = s->conf.macaddr.a[4] | s->conf.macaddr.a[5] << 8;



    s->nic = qemu_new_nic(&net_rtl8139_info, &s->conf,

                          dev->qdev.info->name, dev->qdev.id, s);

    qemu_format_nic_info_str(&s->nic->nc, s->conf.macaddr.a);



    s->cplus_txbuffer = NULL;

    s->cplus_txbuffer_len = 0;

    s->cplus_txbuffer_offset = 0;



    s->TimerExpire = 0;

    s->timer = qemu_new_timer_ns(vm_clock, rtl8139_timer, s);

    rtl8139_set_next_tctr_time(s, qemu_get_clock_ns(vm_clock));



    add_boot_device_path(s->conf.bootindex, &dev->qdev, "/ethernet-phy@0");



    return 0;

}
