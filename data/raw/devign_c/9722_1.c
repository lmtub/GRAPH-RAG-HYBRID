static void usbredir_interrupt_packet(void *priv, uint32_t id,

    struct usb_redir_interrupt_packet_header *interrupt_packet,

    uint8_t *data, int data_len)

{

    USBRedirDevice *dev = priv;

    uint8_t ep = interrupt_packet->endpoint;



    DPRINTF("interrupt-in status %d ep %02X len %d id %u\n",

            interrupt_packet->status, ep, data_len, id);



    if (dev->endpoint[EP2I(ep)].type != USB_ENDPOINT_XFER_INT) {

        ERROR("received int packet for non interrupt endpoint %02X\n", ep);

        free(data);

        return;

    }



    if (ep & USB_DIR_IN) {

        if (dev->endpoint[EP2I(ep)].interrupt_started == 0) {

            DPRINTF("received int packet while not started ep %02X\n", ep);

            free(data);

            return;

        }



        /* bufp_alloc also adds the packet to the ep queue */

        bufp_alloc(dev, data, data_len, interrupt_packet->status, ep);

    } else {

        int len = interrupt_packet->length;



        AsyncURB *aurb = async_find(dev, id);

        if (!aurb) {

            return;

        }



        if (aurb->interrupt_packet.endpoint != interrupt_packet->endpoint) {

            ERROR("return int packet mismatch, please report this!\n");

            len = USB_RET_NAK;

        }



        if (aurb->packet) {

            aurb->packet->len = usbredir_handle_status(dev,

                                               interrupt_packet->status, len);

            usb_packet_complete(&dev->dev, aurb->packet);

        }

        async_free(dev, aurb);

    }

}
