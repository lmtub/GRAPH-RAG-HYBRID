static void i6300esb_pc_init(PCIBus *pci_bus)

{

    I6300State *d;

    uint8_t *pci_conf;



    if (!pci_bus) {

        fprintf(stderr, "wdt_i6300esb: no PCI bus in this machine\n");

        return;

    }



    d = (I6300State *)

        pci_register_device (pci_bus, "i6300esb_wdt", sizeof (I6300State),

                             -1,

                             i6300esb_config_read, i6300esb_config_write);



    d->reboot_enabled = 1;

    d->clock_scale = CLOCK_SCALE_1KHZ;

    d->int_type = INT_TYPE_IRQ;

    d->free_run = 0;

    d->locked = 0;

    d->enabled = 0;

    d->timer = qemu_new_timer(vm_clock, i6300esb_timer_expired, d);

    d->timer1_preload = 0xfffff;

    d->timer2_preload = 0xfffff;

    d->stage = 1;

    d->unlock_state = 0;

    d->previous_reboot_flag = 0;



    pci_conf = d->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_INTEL_ESB_9);

    pci_config_set_class(pci_conf, PCI_CLASS_SYSTEM_OTHER);

    pci_conf[0x0e] = 0x00;



    pci_register_bar(&d->dev, 0, 0x10,

                            PCI_ADDRESS_SPACE_MEM, i6300esb_map);



    register_savevm("i6300esb_wdt", -1, sizeof(I6300State),

                     i6300esb_save, i6300esb_load, d);

}
