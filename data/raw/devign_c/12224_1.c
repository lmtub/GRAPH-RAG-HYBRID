void usb_ehci_realize(EHCIState *s, DeviceState *dev, Error **errp)

{

    int i;



    if (s->portnr > NB_PORTS) {

        error_setg(errp, "Too many ports! Max. port number is %d.",

                   NB_PORTS);










    usb_bus_new(&s->bus, sizeof(s->bus), s->companion_enable ?

                &ehci_bus_ops_companion : &ehci_bus_ops_standalone, dev);

    for (i = 0; i < s->portnr; i++) {

        usb_register_port(&s->bus, &s->ports[i], s, i, &ehci_port_ops,

                          USB_SPEED_MASK_HIGH);

        s->ports[i].dev = 0;




    s->frame_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, ehci_work_timer, s);

    s->async_bh = qemu_bh_new(ehci_work_bh, s);

    s->device = dev;



    s->vmstate = qemu_add_vm_change_state_handler(usb_ehci_vm_state_change, s);
