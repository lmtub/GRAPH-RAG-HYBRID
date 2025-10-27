static int pci_ne2000_init(PCIDevice *pci_dev)

{

    PCINE2000State *d = DO_UPCAST(PCINE2000State, dev, pci_dev);

    NE2000State *s;

    uint8_t *pci_conf;



    pci_conf = d->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_REALTEK);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_REALTEK_8029);

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    /* TODO: RST# value should be 0. PCI spec 6.2.4 */

    pci_conf[PCI_INTERRUPT_PIN] = 1; // interrupt pin 0



    pci_register_bar(&d->dev, 0, 0x100,

                           PCI_BASE_ADDRESS_SPACE_IO, ne2000_map);

    s = &d->ne2000;

    s->irq = d->dev.irq[0];



    qemu_macaddr_default_if_unset(&s->c.macaddr);

    ne2000_reset(s);



    s->nic = qemu_new_nic(&net_ne2000_info, &s->c,

                          pci_dev->qdev.info->name, pci_dev->qdev.id, s);

    qemu_format_nic_info_str(&s->nic->nc, s->c.macaddr.a);



    if (!pci_dev->qdev.hotplugged) {

        static int loaded = 0;

        if (!loaded) {

            rom_add_option("pxe-ne2k_pci.rom", -1);

            loaded = 1;

        }

    }



    add_boot_device_path(s->c.bootindex, &pci_dev->qdev, "/ethernet-phy@0");



    return 0;

}
