static USBDevice *usb_try_create_simple(USBBus *bus, const char *name,

                                        Error **errp)

{

    Error *err = NULL;

    USBDevice *dev;



    dev = USB_DEVICE(qdev_try_create(&bus->qbus, name));

    if (!dev) {

        error_setg(errp, "Failed to create USB device '%s'", name);

        return NULL;

    }

    object_property_set_bool(OBJECT(dev), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

        error_prepend(errp, "Failed to initialize USB device '%s': ",

                      name);

        object_unparent(OBJECT(dev));

        return NULL;

    }

    return dev;

}
