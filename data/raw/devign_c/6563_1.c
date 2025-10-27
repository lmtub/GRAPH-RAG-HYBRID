static int usbredir_handle_interrupt_data(USBRedirDevice *dev,

                                           USBPacket *p, uint8_t ep)

{

    if (ep & USB_DIR_IN) {

        /* Input interrupt endpoint, buffered packet input */

        struct buf_packet *intp;

        int status, len;



        if (!dev->endpoint[EP2I(ep)].interrupt_started &&

                !dev->endpoint[EP2I(ep)].interrupt_error) {

            struct usb_redir_start_interrupt_receiving_header start_int = {

                .endpoint = ep,

            };

            /* No id, we look at the ep when receiving a status back */

            usbredirparser_send_start_interrupt_receiving(dev->parser, 0,

                                                          &start_int);

            usbredirparser_do_write(dev->parser);

            DPRINTF("interrupt recv started ep %02X\n", ep);

            dev->endpoint[EP2I(ep)].interrupt_started = 1;

        }



        intp = QTAILQ_FIRST(&dev->endpoint[EP2I(ep)].bufpq);

        if (intp == NULL) {

            DPRINTF2("interrupt-token-in ep %02X, no intp\n", ep);

            /* Check interrupt_error for stream errors */

            status = dev->endpoint[EP2I(ep)].interrupt_error;

            dev->endpoint[EP2I(ep)].interrupt_error = 0;

            return usbredir_handle_status(dev, status, 0);

        }

        DPRINTF("interrupt-token-in ep %02X status %d len %d\n", ep,

                intp->status, intp->len);



        status = intp->status;

        if (status != usb_redir_success) {

            bufp_free(dev, intp, ep);

            return usbredir_handle_status(dev, status, 0);

        }



        len = intp->len;

        if (len > p->len) {

            ERROR("received int data is larger then packet ep %02X\n", ep);

            bufp_free(dev, intp, ep);

            return USB_RET_NAK;

        }

        memcpy(p->data, intp->data, len);

        bufp_free(dev, intp, ep);

        return len;

    } else {

        /* Output interrupt endpoint, normal async operation */

        AsyncURB *aurb = async_alloc(dev, p);

        struct usb_redir_interrupt_packet_header interrupt_packet;



        DPRINTF("interrupt-out ep %02X len %d id %u\n", ep, p->len,

                aurb->packet_id);



        interrupt_packet.endpoint  = ep;

        interrupt_packet.length    = p->len;

        aurb->interrupt_packet     = interrupt_packet;



        usbredir_log_data(dev, "interrupt data out:", p->data, p->len);

        usbredirparser_send_interrupt_packet(dev->parser, aurb->packet_id,

                                        &interrupt_packet, p->data, p->len);

        usbredirparser_do_write(dev->parser);

        return USB_RET_ASYNC;

    }

}
