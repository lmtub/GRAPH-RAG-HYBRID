static int vt82c686b_pm_initfn(PCIDevice *dev)

{

    VT686PMState *s = DO_UPCAST(VT686PMState, dev, dev);

    uint8_t *pci_conf;



    pci_conf = s->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_VIA);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_VIA_ACPI);

    pci_config_set_class(pci_conf, PCI_CLASS_BRIDGE_OTHER);

    pci_config_set_revision(pci_conf, 0x40);



    pci_set_word(pci_conf + PCI_COMMAND, 0);

    pci_set_word(pci_conf + PCI_STATUS, PCI_STATUS_FAST_BACK |

                 PCI_STATUS_DEVSEL_MEDIUM);



    /* 0x48-0x4B is Power Management I/O Base */

    pci_set_long(pci_conf + 0x48, 0x00000001);



    /* SMB ports:0xeee0~0xeeef */

    s->smb_io_base =((s->smb_io_base & 0xfff0) + 0x0);

    pci_conf[0x90] = s->smb_io_base | 1;

    pci_conf[0x91] = s->smb_io_base >> 8;

    pci_conf[0xd2] = 0x90;

    register_ioport_write(s->smb_io_base, 0xf, 1, smb_ioport_writeb, &s->smb);

    register_ioport_read(s->smb_io_base, 0xf, 1, smb_ioport_readb, &s->smb);



    apm_init(&s->apm, NULL, s);



    acpi_pm_tmr_init(&s->tmr, pm_tmr_timer);

    acpi_pm1_cnt_init(&s->pm1_cnt, NULL);



    pm_smbus_init(&s->dev.qdev, &s->smb);



    return 0;

}
