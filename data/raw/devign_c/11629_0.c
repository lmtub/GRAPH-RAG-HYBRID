static void xhci_port_reset(XHCIPort *port)

{

    trace_usb_xhci_port_reset(port->portnr);



    if (!xhci_port_have_device(port)) {

        return;

    }



    usb_device_reset(port->uport->dev);



    switch (port->uport->dev->speed) {

    case USB_SPEED_LOW:

    case USB_SPEED_FULL:

    case USB_SPEED_HIGH:

        set_field(&port->portsc, PLS_U0, PORTSC_PLS);

        trace_usb_xhci_port_link(port->portnr, PLS_U0);

        port->portsc |= PORTSC_PED;

        break;

    }



    port->portsc &= ~PORTSC_PR;

    xhci_port_notify(port, PORTSC_PRC);

}
