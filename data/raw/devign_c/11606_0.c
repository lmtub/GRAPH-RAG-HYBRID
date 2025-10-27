static int uhci_broadcast_packet(UHCIState *s, USBPacket *p)

{

    UHCIPort *port;

    USBDevice *dev;

    int i, ret;



#ifdef DEBUG_PACKET

    {

        const char *pidstr;

        switch(p->pid) {

        case USB_TOKEN_SETUP: pidstr = "SETUP"; break;

        case USB_TOKEN_IN: pidstr = "IN"; break;

        case USB_TOKEN_OUT: pidstr = "OUT"; break;

        default: pidstr = "?"; break;

        }

        printf("frame %d: pid=%s addr=0x%02x ep=%d len=%d\n",

               s->frnum, pidstr, p->devaddr, p->devep, p->len);

        if (p->pid != USB_TOKEN_IN) {

            printf("     data_out=");

            for(i = 0; i < p->len; i++) {

                printf(" %02x", p->data[i]);

            }

            printf("\n");

        }

    }

#endif

    for(i = 0; i < NB_PORTS; i++) {

        port = &s->ports[i];

        dev = port->port.dev;

        if (dev && (port->ctrl & UHCI_PORT_EN)) {

            ret = dev->handle_packet(dev, p);

            if (ret != USB_RET_NODEV) {

#ifdef DEBUG_PACKET

                if (ret == USB_RET_ASYNC) {

                    printf("usb-uhci: Async packet\n");

                } else {

                    printf("     ret=%d ", ret);

                    if (p->pid == USB_TOKEN_IN && ret > 0) {

                        printf("data_in=");

                        for(i = 0; i < ret; i++) {

                            printf(" %02x", p->data[i]);

                        }

                    }

                    printf("\n");

                }

#endif

                return ret;

            }

        }

    }

    return USB_RET_NODEV;

}
