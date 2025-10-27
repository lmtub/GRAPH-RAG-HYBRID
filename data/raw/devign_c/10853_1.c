static int uhci_broadcast_packet(UHCIState *s, USBPacket *p)

{

    int i, ret;



    DPRINTF("uhci: packet enter. pid %s addr 0x%02x ep %d len %d\n",

           pid2str(p->pid), p->devaddr, p->devep, p->len);

    if (p->pid == USB_TOKEN_OUT || p->pid == USB_TOKEN_SETUP)

        dump_data(p->data, p->len);



    ret = USB_RET_NODEV;

    for (i = 0; i < NB_PORTS && ret == USB_RET_NODEV; i++) {

        UHCIPort *port = &s->ports[i];

        USBDevice *dev = port->port.dev;



        if (dev && (port->ctrl & UHCI_PORT_EN))

            ret = usb_handle_packet(dev, p);

    }



    DPRINTF("uhci: packet exit. ret %d len %d\n", ret, p->len);

    if (p->pid == USB_TOKEN_IN && ret > 0)

        dump_data(p->data, ret);



    return ret;

}
