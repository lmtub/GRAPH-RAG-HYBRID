static void e100_pci_reset(EEPRO100State * s, E100PCIDeviceInfo *e100_device)

{

    uint32_t device = s->device;

    uint8_t *pci_conf = s->dev.config;



    TRACE(OTHER, logout("%p\n", s));



    /* PCI Vendor ID */

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    /* PCI Device ID */

    pci_config_set_device_id(pci_conf, e100_device->device_id);

    /* PCI Status */

    pci_set_word(pci_conf + PCI_STATUS, PCI_STATUS_DEVSEL_MEDIUM |

                                        PCI_STATUS_FAST_BACK);

    /* PCI Revision ID */

    pci_config_set_revision(pci_conf, e100_device->revision);

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    /* PCI Latency Timer */

    pci_set_byte(pci_conf + PCI_LATENCY_TIMER, 0x20);   /* latency timer = 32 clocks */

    /* Capability Pointer is set by PCI framework. */

    /* Interrupt Line */

    /* Interrupt Pin */

    pci_set_byte(pci_conf + PCI_INTERRUPT_PIN, 1);      /* interrupt pin A */

    /* Minimum Grant */

    pci_set_byte(pci_conf + PCI_MIN_GNT, 0x08);

    /* Maximum Latency */

    pci_set_byte(pci_conf + PCI_MAX_LAT, 0x18);



    s->stats_size = e100_device->stats_size;

    s->has_extended_tcb_support = e100_device->has_extended_tcb_support;



    switch (device) {

    case i82550:

    case i82551:

    case i82557A:

    case i82557B:

    case i82557C:

    case i82558A:

    case i82558B:

    case i82559A:

    case i82559B:

    case i82559ER:

    case i82562:

    case i82801:

        break;

    case i82559C:

#if EEPROM_SIZE > 0

        pci_set_word(pci_conf + PCI_SUBSYSTEM_VENDOR_ID, PCI_VENDOR_ID_INTEL);

        pci_set_word(pci_conf + PCI_SUBSYSTEM_ID, 0x0040);

#endif

        break;

    default:

        logout("Device %X is undefined!\n", device);

    }



    /* Standard TxCB. */

    s->configuration[6] |= BIT(4);



    /* Standard statistical counters. */

    s->configuration[6] |= BIT(5);



    if (s->stats_size == 80) {

        /* TODO: check TCO Statistical Counters bit. Documentation not clear. */

        if (s->configuration[6] & BIT(2)) {

            /* TCO statistical counters. */

            assert(s->configuration[6] & BIT(5));

        } else {

            if (s->configuration[6] & BIT(5)) {

                /* No extended statistical counters, i82557 compatible. */

                s->stats_size = 64;

            } else {

                /* i82558 compatible. */

                s->stats_size = 76;

            }

        }

    } else {

        if (s->configuration[6] & BIT(5)) {

            /* No extended statistical counters. */

            s->stats_size = 64;

        }

    }

    assert(s->stats_size > 0 && s->stats_size <= sizeof(s->statistics));



    if (e100_device->power_management) {

        /* Power Management Capabilities */

        int cfg_offset = 0xdc;

        int r = pci_add_capability(&s->dev, PCI_CAP_ID_PM,

                                   cfg_offset, PCI_PM_SIZEOF);

        assert(r >= 0);

        pci_set_word(pci_conf + cfg_offset + PCI_PM_PMC, 0x7e21);

#if 0 /* TODO: replace dummy code for power management emulation. */

        /* TODO: Power Management Control / Status. */

        pci_set_word(pci_conf + cfg_offset + PCI_PM_CTRL, 0x0000);

        /* TODO: Ethernet Power Consumption Registers (i82559 and later). */

        pci_set_byte(pci_conf + cfg_offset + PCI_PM_PPB_EXTENSIONS, 0x0000);

#endif

    }



#if EEPROM_SIZE > 0

    if (device == i82557C || device == i82558B || device == i82559C) {

        /*

        TODO: get vendor id from EEPROM for i82557C or later.

        TODO: get device id from EEPROM for i82557C or later.

        TODO: status bit 4 can be disabled by EEPROM for i82558, i82559.

        TODO: header type is determined by EEPROM for i82559.

        TODO: get subsystem id from EEPROM for i82557C or later.

        TODO: get subsystem vendor id from EEPROM for i82557C or later.

        TODO: exp. rom baddr depends on a bit in EEPROM for i82558 or later.

        TODO: capability pointer depends on EEPROM for i82558.

        */

        logout("Get device id and revision from EEPROM!!!\n");

    }

#endif /* EEPROM_SIZE > 0 */

}
