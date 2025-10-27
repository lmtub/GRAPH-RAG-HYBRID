static int usb_hub_initfn(USBDevice *dev)

{

    USBHubState *s = DO_UPCAST(USBHubState, dev, dev);

    USBHubPort *port;

    int i;



    s->dev.speed = USB_SPEED_FULL;

    for (i = 0; i < NUM_PORTS; i++) {

        port = &s->ports[i];

        usb_register_port(usb_bus_from_device(dev),

                          &port->port, s, i, &s->dev, usb_hub_attach);

        port->wPortStatus = PORT_STAT_POWER;

        port->wPortChange = 0;

    }

    return 0;

}
