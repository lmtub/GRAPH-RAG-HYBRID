void usb_wakeup(USBEndpoint *ep, unsigned int stream)

{

    USBDevice *dev = ep->dev;

    USBBus *bus = usb_bus_from_device(dev);











    if (dev->remote_wakeup && dev->port && dev->port->ops->wakeup) {

        dev->port->ops->wakeup(dev->port);


    if (bus->ops->wakeup_endpoint) {

        bus->ops->wakeup_endpoint(bus, ep, stream);

