static void xhci_port_update(XHCIPort *port, int is_detach)

{

    port->portsc = PORTSC_PP;

    if (port->uport->dev && port->uport->dev->attached && !is_detach &&

        (1 << port->uport->dev->speed) & port->speedmask) {

        port->portsc |= PORTSC_CCS;

        switch (port->uport->dev->speed) {

        case USB_SPEED_LOW:

            port->portsc |= PORTSC_SPEED_LOW;

            break;

        case USB_SPEED_FULL:

            port->portsc |= PORTSC_SPEED_FULL;

            break;

        case USB_SPEED_HIGH:

            port->portsc |= PORTSC_SPEED_HIGH;

            break;

        case USB_SPEED_SUPER:

            port->portsc |= PORTSC_SPEED_SUPER;

            break;

        }

    }



    if (xhci_running(port->xhci)) {

        port->portsc |= PORTSC_CSC;

        XHCIEvent ev = { ER_PORT_STATUS_CHANGE, CC_SUCCESS,

                         port->portnr << 24};

        xhci_event(port->xhci, &ev, 0);

        DPRINTF("xhci: port change event for port %d\n", port->portnr);

    }

}
