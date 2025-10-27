static void usb_host_speed_compat(USBHostDevice *s)

{

    USBDevice *udev = USB_DEVICE(s);

    struct libusb_config_descriptor *conf;

    const struct libusb_interface_descriptor *intf;

    const struct libusb_endpoint_descriptor *endp;

#ifdef HAVE_STREAMS

    struct libusb_ss_endpoint_companion_descriptor *endp_ss_comp;

#endif

    bool compat_high = true;

    bool compat_full = true;

    uint8_t type;

    int rc, c, i, a, e;



    for (c = 0;; c++) {

        rc = libusb_get_config_descriptor(s->dev, c, &conf);

        if (rc != 0) {

            break;

        }

        for (i = 0; i < conf->bNumInterfaces; i++) {

            for (a = 0; a < conf->interface[i].num_altsetting; a++) {

                intf = &conf->interface[i].altsetting[a];

                for (e = 0; e < intf->bNumEndpoints; e++) {

                    endp = &intf->endpoint[e];

                    type = endp->bmAttributes & 0x3;

                    switch (type) {

                    case 0x01: /* ISO */

                        compat_full = false;

                        compat_high = false;

                        break;

                    case 0x02: /* BULK */

#ifdef HAVE_STREAMS

                        rc = libusb_get_ss_endpoint_companion_descriptor

                            (ctx, endp, &endp_ss_comp);

                        if (rc == LIBUSB_SUCCESS) {

                            libusb_free_ss_endpoint_companion_descriptor

                                (endp_ss_comp);

                            compat_full = false;

                            compat_high = false;

                        }

#endif

                        break;

                    case 0x03: /* INTERRUPT */

                        if (endp->wMaxPacketSize > 64) {

                            compat_full = false;

                        }

                        if (endp->wMaxPacketSize > 1024) {

                            compat_high = false;

                        }

                        break;

                    }

                }

            }

        }

        libusb_free_config_descriptor(conf);

    }



    udev->speedmask = (1 << udev->speed);

    if (udev->speed == USB_SPEED_SUPER && compat_high) {

        udev->speedmask |= USB_SPEED_MASK_HIGH;

    }

    if (udev->speed == USB_SPEED_SUPER && compat_full) {

        udev->speedmask |= USB_SPEED_MASK_FULL;

    }

    if (udev->speed == USB_SPEED_HIGH && compat_full) {

        udev->speedmask |= USB_SPEED_MASK_FULL;

    }

}
