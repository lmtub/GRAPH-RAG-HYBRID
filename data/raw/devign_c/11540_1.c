static int usbredir_handle_iso_data(USBRedirDevice *dev, USBPacket *p,

                                     uint8_t ep)

{

    int status, len;



    if (!dev->endpoint[EP2I(ep)].iso_started &&

            !dev->endpoint[EP2I(ep)].iso_error) {

        struct usb_redir_start_iso_stream_header start_iso = {

            .endpoint = ep,

            /* TODO maybe do something with these depending on ep interval? */

            .pkts_per_urb = 32,

            .no_urbs = 3,

        };

        /* No id, we look at the ep when receiving a status back */

        usbredirparser_send_start_iso_stream(dev->parser, 0, &start_iso);

        usbredirparser_do_write(dev->parser);

        DPRINTF("iso stream started ep %02X\n", ep);

        dev->endpoint[EP2I(ep)].iso_started = 1;

    }



    if (ep & USB_DIR_IN) {

        struct buf_packet *isop;



        isop = QTAILQ_FIRST(&dev->endpoint[EP2I(ep)].bufpq);

        if (isop == NULL) {

            DPRINTF2("iso-token-in ep %02X, no isop\n", ep);

            /* Check iso_error for stream errors, otherwise its an underrun */

            status = dev->endpoint[EP2I(ep)].iso_error;

            dev->endpoint[EP2I(ep)].iso_error = 0;

            return usbredir_handle_status(dev, status, 0);

        }

        DPRINTF2("iso-token-in ep %02X status %d len %d\n", ep, isop->status,

                 isop->len);



        status = isop->status;

        if (status != usb_redir_success) {

            bufp_free(dev, isop, ep);

            return usbredir_handle_status(dev, status, 0);

        }



        len = isop->len;

        if (len > p->len) {

            ERROR("received iso data is larger then packet ep %02X\n", ep);

            bufp_free(dev, isop, ep);

            return USB_RET_NAK;

        }

        memcpy(p->data, isop->data, len);

        bufp_free(dev, isop, ep);

        return len;

    } else {

        /* If the stream was not started because of a pending error don't

           send the packet to the usb-host */

        if (dev->endpoint[EP2I(ep)].iso_started) {

            struct usb_redir_iso_packet_header iso_packet = {

                .endpoint = ep,

                .length = p->len

            };

            /* No id, we look at the ep when receiving a status back */

            usbredirparser_send_iso_packet(dev->parser, 0, &iso_packet,

                                           p->data, p->len);

            usbredirparser_do_write(dev->parser);

        }

        status = dev->endpoint[EP2I(ep)].iso_error;

        dev->endpoint[EP2I(ep)].iso_error = 0;

        DPRINTF2("iso-token-out ep %02X status %d len %d\n", ep, status,

                 p->len);

        return usbredir_handle_status(dev, status, p->len);

    }

}
