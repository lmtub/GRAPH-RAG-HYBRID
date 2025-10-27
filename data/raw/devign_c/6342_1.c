void usb_desc_create_serial(USBDevice *dev)

{

    DeviceState *hcd = dev->qdev.parent_bus->parent;

    const USBDesc *desc = usb_device_get_usb_desc(dev);

    int index = desc->id.iSerialNumber;

    char serial[64];

    char *path;

    int dst;



    if (dev->serial) {

        /* 'serial' usb bus property has priority if present */

        usb_desc_set_string(dev, index, dev->serial);

        return;

    }



    assert(index != 0 && desc->str[index] != NULL);

    dst = snprintf(serial, sizeof(serial), "%s", desc->str[index]);

    path = qdev_get_dev_path(hcd);

    if (path) {

        dst += snprintf(serial+dst, sizeof(serial)-dst, "-%s", path);

    }

    dst += snprintf(serial+dst, sizeof(serial)-dst, "-%s", dev->port->path);

    usb_desc_set_string(dev, index, serial);


}