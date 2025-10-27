int usb_device_delete_addr(int busnr, int addr)

{

    USBBus *bus;

    USBPort *port;

    USBDevice *dev;



    bus = usb_bus_find(busnr);

    if (!bus)

        return -1;



    TAILQ_FOREACH(port, &bus->used, next) {

        if (port->dev->addr == addr)

            break;

    }

    if (!port)

        return -1;



    dev = port->dev;

    TAILQ_REMOVE(&bus->used, port, next);

    bus->nused--;



    usb_attach(port, NULL);

    dev->info->handle_destroy(dev);



    TAILQ_INSERT_TAIL(&bus->free, port, next);

    bus->nfree++;

    return 0;

}
