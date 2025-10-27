static int usb_host_claim_interfaces(USBHostDevice *s, int configuration)

{

    USBDevice *udev = USB_DEVICE(s);

    struct libusb_config_descriptor *conf;

    int rc, i;



    for (i = 0; i < USB_MAX_INTERFACES; i++) {

        udev->altsetting[i] = 0;

    }

    udev->ninterfaces   = 0;

    udev->configuration = 0;



    if (configuration == 0) {

        /* address state - ignore */

        return USB_RET_SUCCESS;

    }



    usb_host_detach_kernel(s);



    rc = libusb_get_active_config_descriptor(s->dev, &conf);

    if (rc != 0) {

        return USB_RET_STALL;

    }



    for (i = 0; i < conf->bNumInterfaces; i++) {

        trace_usb_host_claim_interface(s->bus_num, s->addr, configuration, i);

        rc = libusb_claim_interface(s->dh, i);

        usb_host_libusb_error("libusb_claim_interface", rc);

        if (rc != 0) {

            return USB_RET_STALL;

        }

        s->ifs[i].claimed = true;

    }



    udev->ninterfaces   = conf->bNumInterfaces;

    udev->configuration = configuration;



    libusb_free_config_descriptor(conf);

    return USB_RET_SUCCESS;

}
