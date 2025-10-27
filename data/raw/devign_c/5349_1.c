USBDevice *usb_host_device_open(USBBus *bus, const char *devname)

{

    struct USBAutoFilter filter;

    USBDevice *dev;

    char *p;



    dev = usb_create(bus, "usb-host");



    if (strstr(devname, "auto:")) {

        if (parse_filter(devname, &filter) < 0) {

            goto fail;

        }

    } else {

        p = strchr(devname, '.');

        if (p) {

            filter.bus_num    = strtoul(devname, NULL, 0);

            filter.addr       = strtoul(p + 1, NULL, 0);

            filter.vendor_id  = 0;

            filter.product_id = 0;

        } else {

            p = strchr(devname, ':');

            if (p) {

                filter.bus_num    = 0;

                filter.addr       = 0;

                filter.vendor_id  = strtoul(devname, NULL, 16);

                filter.product_id = strtoul(p + 1, NULL, 16);

            } else {

                goto fail;

            }

        }

    }



    qdev_prop_set_uint32(&dev->qdev, "hostbus",   filter.bus_num);

    qdev_prop_set_uint32(&dev->qdev, "hostaddr",  filter.addr);

    qdev_prop_set_uint32(&dev->qdev, "vendorid",  filter.vendor_id);

    qdev_prop_set_uint32(&dev->qdev, "productid", filter.product_id);

    qdev_init_nofail(&dev->qdev);

    return dev;



fail:

    object_unparent(OBJECT(dev));

    return NULL;

}
