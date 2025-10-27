static void usb_host_realize(USBDevice *udev, Error **errp)

{

    USBHostDevice *s = USB_HOST_DEVICE(udev);



    if (s->match.vendor_id > 0xffff) {

        error_setg(errp, "vendorid out of range");

        return;

    }

    if (s->match.product_id > 0xffff) {

        error_setg(errp, "productid out of range");

        return;

    }

    if (s->match.addr > 127) {

        error_setg(errp, "hostaddr out of range");

        return;

    }



    loglevel = s->loglevel;

    udev->flags |= (1 << USB_DEV_FLAG_IS_HOST);

    udev->auto_attach = 0;

    QTAILQ_INIT(&s->requests);

    QTAILQ_INIT(&s->isorings);



    s->exit.notify = usb_host_exit_notifier;

    qemu_add_exit_notifier(&s->exit);



    QTAILQ_INSERT_TAIL(&hostdevs, s, next);

    usb_host_auto_check(NULL);

}
