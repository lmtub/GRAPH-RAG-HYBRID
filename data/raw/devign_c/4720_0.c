void usb_register_port(USBBus *bus, USBPort *port, void *opaque, int index,

                       usb_attachfn attach)

{

    port->opaque = opaque;

    port->index = index;

    port->attach = attach;

    TAILQ_INSERT_TAIL(&bus->free, port, next);

    bus->nfree++;

}
