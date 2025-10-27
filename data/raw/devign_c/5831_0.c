int usb_device_detach(USBDevice *dev)

{

    USBBus *bus = usb_bus_from_device(dev);

    USBPort *port;



    if (!dev->attached) {

        error_report("Error: tried to detach unattached usb device %s\n",

                dev->product_desc);

        return -1;

    }

    dev->attached--;



    QTAILQ_FOREACH(port, &bus->used, next) {

        if (port->dev == dev)

            break;

    }

    assert(port != NULL);



    QTAILQ_REMOVE(&bus->used, port, next);

    bus->nused--;



    usb_attach(port, NULL);



    QTAILQ_INSERT_TAIL(&bus->free, port, next);

    bus->nfree++;

    return 0;

}
