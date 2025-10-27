void usb_device_attach(USBDevice *dev, Error **errp)

{

    USBBus *bus = usb_bus_from_device(dev);

    USBPort *port = dev->port;

    char devspeed[32], portspeed[32];



    assert(port != NULL);

    assert(!dev->attached);

    usb_mask_to_str(devspeed, sizeof(devspeed), dev->speedmask);

    usb_mask_to_str(portspeed, sizeof(portspeed), port->speedmask);

    trace_usb_port_attach(bus->busnr, port->path,

                          devspeed, portspeed);



    if (!(port->speedmask & dev->speedmask)) {

        error_setg(errp, "Warning: speed mismatch trying to attach"

                   " usb device \"%s\" (%s speed)"

                   " to bus \"%s\", port \"%s\" (%s speed)",

                   dev->product_desc, devspeed,

                   bus->qbus.name, port->path, portspeed);

        return;

    }



    dev->attached++;

    usb_attach(port);

}
