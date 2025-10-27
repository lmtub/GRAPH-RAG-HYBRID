pci_e1000_init(PCIBus *bus, NICInfo *nd, int devfn)

{

    E1000State *d;

    uint8_t *pci_conf;

    uint16_t checksum = 0;

    static const char info_str[] = "e1000";

    int i;



    d = (E1000State *)pci_register_device(bus, "e1000",

                sizeof(E1000State), devfn, NULL, NULL);



    pci_conf = d->dev.config;

    memset(pci_conf, 0, 256);



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, E1000_DEVID);

    *(uint16_t *)(pci_conf+0x04) = cpu_to_le16(0x0407);

    *(uint16_t *)(pci_conf+0x06) = cpu_to_le16(0x0010);

    pci_conf[0x08] = 0x03;

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    pci_conf[0x0c] = 0x10;



    pci_conf[0x3d] = 1; // interrupt pin 0



    d->mmio_index = cpu_register_io_memory(0, e1000_mmio_read,

            e1000_mmio_write, d);



    pci_register_io_region((PCIDevice *)d, 0, PNPMMIO_SIZE,

                           PCI_ADDRESS_SPACE_MEM, e1000_mmio_map);



    pci_register_io_region((PCIDevice *)d, 1, IOPORT_SIZE,

                           PCI_ADDRESS_SPACE_IO, ioport_map);



    memmove(d->eeprom_data, e1000_eeprom_template,

        sizeof e1000_eeprom_template);

    for (i = 0; i < 3; i++)

        d->eeprom_data[i] = (nd->macaddr[2*i+1]<<8) | nd->macaddr[2*i];

    for (i = 0; i < EEPROM_CHECKSUM_REG; i++)

        checksum += d->eeprom_data[i];

    checksum = (uint16_t) EEPROM_SUM - checksum;

    d->eeprom_data[EEPROM_CHECKSUM_REG] = checksum;



    memset(d->phy_reg, 0, sizeof d->phy_reg);

    memmove(d->phy_reg, phy_reg_init, sizeof phy_reg_init);

    memset(d->mac_reg, 0, sizeof d->mac_reg);

    memmove(d->mac_reg, mac_reg_init, sizeof mac_reg_init);

    d->rxbuf_min_shift = 1;

    memset(&d->tx, 0, sizeof d->tx);



    d->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 e1000_receive, e1000_can_receive, d);

    d->vc->link_status_changed = e1000_set_link_status;



    qemu_format_nic_info_str(d->vc, nd->macaddr);



    register_savevm(info_str, -1, 2, nic_save, nic_load, d);

    d->dev.unregister = pci_e1000_uninit;



    return (PCIDevice *)d;

}
