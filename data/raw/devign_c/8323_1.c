PCIDevice *pci_rtl8139_init(PCIBus *bus, NICInfo *nd, int devfn)

{

    PCIRTL8139State *d;

    RTL8139State *s;

    uint8_t *pci_conf;



    d = (PCIRTL8139State *)pci_register_device(bus,

                                              "RTL8139", sizeof(PCIRTL8139State),

                                              devfn,

                                              NULL, NULL);

    pci_conf = d->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_REALTEK);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_REALTEK_8139);

    pci_conf[0x04] = 0x05; /* command = I/O space, Bus Master */

    pci_conf[0x08] = RTL8139_PCI_REVID; /* PCI revision ID; >=0x20 is for 8139C+ */

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    pci_conf[0x0e] = 0x00; /* header_type */

    pci_conf[0x3d] = 1;    /* interrupt pin 0 */

    pci_conf[0x34] = 0xdc;



    s = &d->rtl8139;



    /* I/O handler for memory-mapped I/O */

    s->rtl8139_mmio_io_addr =

    cpu_register_io_memory(0, rtl8139_mmio_read, rtl8139_mmio_write, s);



    pci_register_io_region(&d->dev, 0, 0x100,

                           PCI_ADDRESS_SPACE_IO,  rtl8139_ioport_map);



    pci_register_io_region(&d->dev, 1, 0x100,

                           PCI_ADDRESS_SPACE_MEM, rtl8139_mmio_map);



    s->pci_dev = (PCIDevice *)d;

    memcpy(s->macaddr, nd->macaddr, 6);

    rtl8139_reset(s);

    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 rtl8139_receive, rtl8139_can_receive, s);



    qemu_format_nic_info_str(s->vc, s->macaddr);



    s->cplus_txbuffer = NULL;

    s->cplus_txbuffer_len = 0;

    s->cplus_txbuffer_offset = 0;



    register_savevm("rtl8139", -1, 4, rtl8139_save, rtl8139_load, s);



#ifdef RTL8139_ONBOARD_TIMER

    s->timer = qemu_new_timer(vm_clock, rtl8139_timer, s);



    qemu_mod_timer(s->timer,

        rtl8139_get_next_tctr_time(s,qemu_get_clock(vm_clock)));

#endif /* RTL8139_ONBOARD_TIMER */

    return (PCIDevice *)d;

}
