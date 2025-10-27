static int usb_linux_update_endp_table(USBHostDevice *s)

{

    uint8_t *descriptors;

    uint8_t devep, type, alt_interface;

    int interface, length, i, ep, pid;

    struct endp_data *epd;



    for (i = 0; i < MAX_ENDPOINTS; i++) {

        s->ep_in[i].type = INVALID_EP_TYPE;

        s->ep_out[i].type = INVALID_EP_TYPE;

    }



    if (s->configuration == 0) {

        /* not configured yet -- leave all endpoints disabled */

        return 0;

    }



    /* get the desired configuration, interface, and endpoint descriptors

     * from device description */

    descriptors = &s->descr[18];

    length = s->descr_len - 18;

    i = 0;



    if (descriptors[i + 1] != USB_DT_CONFIG ||

        descriptors[i + 5] != s->configuration) {

        fprintf(stderr, "invalid descriptor data - configuration %d\n",

                s->configuration);

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

        alt_interface = usb_linux_get_alt_setting(s, s->configuration,

                                                  interface);



        /* the current interface descriptor is the active interface

         * and has endpoints */

        if (descriptors[i + 3] != alt_interface) {

            i += descriptors[i];

            continue;

        }



        /* advance to the endpoints */

        while (i < length && descriptors[i +1] != USB_DT_ENDPOINT) {

            i += descriptors[i];

        }



        if (i >= length)

            break;



        while (i < length) {

            if (descriptors[i + 1] != USB_DT_ENDPOINT) {

                break;

            }



            devep = descriptors[i + 2];

            pid = (devep & USB_DIR_IN) ? USB_TOKEN_IN : USB_TOKEN_OUT;

            ep = devep & 0xf;

            if (ep == 0) {

                fprintf(stderr, "usb-linux: invalid ep descriptor, ep == 0\n");

                return 1;

            }



            switch (descriptors[i + 3] & 0x3) {

            case 0x00:

                type = USBDEVFS_URB_TYPE_CONTROL;

                break;

            case 0x01:

                type = USBDEVFS_URB_TYPE_ISO;

                set_max_packet_size(s, pid, ep, descriptors + i);

                break;

            case 0x02:

                type = USBDEVFS_URB_TYPE_BULK;

                break;

            case 0x03:

                type = USBDEVFS_URB_TYPE_INTERRUPT;

                break;

            default:

                DPRINTF("usb_host: malformed endpoint type\n");

                type = USBDEVFS_URB_TYPE_BULK;

            }

            epd = get_endp(s, pid, ep);

            assert(epd->type == INVALID_EP_TYPE);

            epd->type = type;

            epd->halted = 0;



            i += descriptors[i];

        }

    }

    return 0;

}
