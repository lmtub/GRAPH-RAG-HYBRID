static int usb_qdev_init(DeviceState *qdev)

{

    USBDevice *dev = USB_DEVICE(qdev);

    int rc;



    pstrcpy(dev->product_desc, sizeof(dev->product_desc),

            usb_device_get_product_desc(dev));

    dev->auto_attach = 1;

    QLIST_INIT(&dev->strings);

    usb_ep_init(dev);

    rc = usb_claim_port(dev);

    if (rc != 0) {

        return rc;

    }

    rc = usb_device_init(dev);

    if (rc != 0) {

        usb_release_port(dev);

        return rc;

    }

    if (dev->auto_attach) {

        rc = usb_device_attach(dev);

        if (rc != 0) {

            usb_qdev_exit(qdev);

            return rc;

        }

    }

    return 0;

}
