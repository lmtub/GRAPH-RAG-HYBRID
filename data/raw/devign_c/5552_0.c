USBBus *usb_bus_new(DeviceState *host)

{

    USBBus *bus;



    bus = FROM_QBUS(USBBus, qbus_create(&usb_bus_info, host, NULL));

    bus->busnr = next_usb_bus++;

    TAILQ_INIT(&bus->free);

    TAILQ_INIT(&bus->used);

    TAILQ_INSERT_TAIL(&busses, bus, next);

    return bus;

}
