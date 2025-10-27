static int usb_uhci_common_initfn(UHCIState *s)

{

    uint8_t *pci_conf = s->dev.config;

    int i;



    pci_conf[PCI_REVISION_ID] = 0x01; // revision number

    pci_conf[PCI_CLASS_PROG] = 0x00;

    pci_config_set_class(pci_conf, PCI_CLASS_SERIAL_USB);

    /* TODO: reset value should be 0. */

    pci_conf[PCI_INTERRUPT_PIN] = 4; // interrupt pin 3

    pci_conf[0x60] = 0x10; // release number



    usb_bus_new(&s->bus, &s->dev.qdev);

    for(i = 0; i < NB_PORTS; i++) {

        usb_register_port(&s->bus, &s->ports[i].port, s, i, &uhci_port_ops,

                          USB_SPEED_MASK_LOW | USB_SPEED_MASK_FULL);

        usb_port_location(&s->ports[i].port, NULL, i+1);

    }

    s->frame_timer = qemu_new_timer_ns(vm_clock, uhci_frame_timer, s);

    s->expire_time = qemu_get_clock_ns(vm_clock) +

        (get_ticks_per_sec() / FRAME_TIMER_FREQ);

    s->num_ports_vmstate = NB_PORTS;



    qemu_register_reset(uhci_reset, s);



    /* Use region 4 for consistency with real hardware.  BSD guests seem

       to rely on this.  */

    pci_register_bar(&s->dev, 4, 0x20,

                           PCI_BASE_ADDRESS_SPACE_IO, uhci_map);



    return 0;

}
