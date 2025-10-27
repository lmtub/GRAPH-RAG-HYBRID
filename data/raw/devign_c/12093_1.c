device_init(usb_host_register_devices)



USBDevice *usb_host_device_open(const char *devname)

{

    struct USBAutoFilter filter = { 0, 0, 0, 0 };

    USBDevice *dev;

    USBHostDevice *s;

    char *p;



    dev = usb_create(NULL /* FIXME */, "USB Host Device");

    s = DO_UPCAST(USBHostDevice, dev, dev);



    if (strstr(devname, "auto:")) {

        if (parse_filter(devname+5, &filter) < 0)

            goto fail;

    } else {

        if ((p = strchr(devname, '.'))) {

            filter.bus_num = strtoul(devname, NULL, 0);

            filter.addr    = strtoul(devname, NULL, 0);

        } else if ((p = strchr(devname, ':'))) {

            filter.vendor_id  = strtoul(devname, NULL, 16);

            filter.product_id = strtoul(devname, NULL, 16);

        } else {

            goto fail;

        }

    }



    qdev_prop_set_uint32(&dev->qdev, "bus",       filter.bus_num);

    qdev_prop_set_uint32(&dev->qdev, "addr",      filter.addr);

    qdev_prop_set_uint32(&dev->qdev, "vendorid",  filter.vendor_id);

    qdev_prop_set_uint32(&dev->qdev, "productid", filter.product_id);

    qdev_init(&dev->qdev);

    return dev;



fail:

    qdev_free(&dev->qdev);

    return NULL;

}
