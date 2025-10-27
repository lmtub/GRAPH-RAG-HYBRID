static void uhci_reset(void *opaque)

{

    UHCIState *s = opaque;

    uint8_t *pci_conf;

    int i;

    UHCIPort *port;



    DPRINTF("uhci: full reset\n");



    pci_conf = s->dev.config;



    pci_conf[0x6a] = 0x01; /* usb clock */

    pci_conf[0x6b] = 0x00;

    s->cmd = 0;

    s->status = 0;

    s->status2 = 0;

    s->intr = 0;

    s->fl_base_addr = 0;

    s->sof_timing = 64;



    for(i = 0; i < NB_PORTS; i++) {

        port = &s->ports[i];

        port->ctrl = 0x0080;

        if (port->port.dev) {

            usb_attach(&port->port, port->port.dev);

        }

    }



    uhci_async_cancel_all(s);

}
