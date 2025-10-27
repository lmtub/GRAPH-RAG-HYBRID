static int usb_linux_update_endp_table(USBHostDevice *s)

{

    uint8_t *descriptors;

    uint8_t devep, type, configuration, alt_interface;

    struct usb_ctrltransfer ct;

    int interface, ret, length, i;



    ct.bRequestType = USB_DIR_IN;

    ct.bRequest = USB_REQ_GET_CONFIGURATION;

    ct.wValue = 0;

    ct.wIndex = 0;

    ct.wLength = 1;

    ct.data = &configuration;

    ct.timeout = 50;



    ret = ioctl(s->fd, USBDEVFS_CONTROL, &ct);

    if (ret < 0) {

        perror("usb_linux_update_endp_table");

        return 1;

    }



    /* in address state */

    if (configuration == 0)

        return 1;



    /* get the desired configuration, interface, and endpoint descriptors

     * from device description */

    descriptors = &s->descr[18];

    length = s->descr_len - 18;

    i = 0;



    if (descriptors[i + 1] != USB_DT_CONFIG ||

        descriptors[i + 5] != configuration) {

        DPRINTF("invalid descriptor data - configuration\n");

        return 1;

    }

    i += descriptors[i];



    while (i < length) {

        if (descriptors[i + 1] != USB_DT_INTERFACE ||

            (descriptors[i + 1] == USB_DT_INTERFACE &&

             descriptors[i + 4] == 0)) {

            i += descriptors[i];

            continue;

        }



        interface = descriptors[i + 2];



        ct.bRequestType = USB_DIR_IN | USB_RECIP_INTERFACE;

        ct.bRequest = USB_REQ_GET_INTERFACE;

        ct.wValue = 0;

        ct.wIndex = interface;

        ct.wLength = 1;

        ct.data = &alt_interface;

        ct.timeout = 50;



        ret = ioctl(s->fd, USBDEVFS_CONTROL, &ct);

        if (ret < 0) {

            alt_interface = interface;

        }



        /* the current interface descriptor is the active interface

         * and has endpoints */

        if (descriptors[i + 3] != alt_interface) {

            i += descriptors[i];

            continue;

        }



        /* advance to the endpoints */

        while (i < length && descriptors[i +1] != USB_DT_ENDPOINT)

            i += descriptors[i];



        if (i >= length)

            break;



        while (i < length) {

            if (descriptors[i + 1] != USB_DT_ENDPOINT)

                break;



            devep = descriptors[i + 2];

            switch (descriptors[i + 3] & 0x3) {

            case 0x00:

                type = USBDEVFS_URB_TYPE_CONTROL;

                break;

            case 0x01:

                type = USBDEVFS_URB_TYPE_ISO;

                break;

            case 0x02:


                break;

            case 0x03:

                type = USBDEVFS_URB_TYPE_INTERRUPT;

                break;




            }

            s->endp_table[(devep & 0xf) - 1].type = type;

            s->endp_table[(devep & 0xf) - 1].halted = 0;



            i += descriptors[i];

        }

    }

    return 0;

}