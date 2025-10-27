static void usbredir_interrupt_packet(void *priv, uint64_t id,

    struct usb_redir_interrupt_packet_header *interrupt_packet,

    uint8_t *data, int data_len)

{

    USBRedirDevice *dev = priv;

    uint8_t ep = interrupt_packet->endpoint;



    DPRINTF("interrupt-in status %d ep %02X len %d id %"PRIu64"\n",

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

        USBPacket *p = usbredir_find_packet_by_id(dev, ep, id);

        if (p) {

            usbredir_handle_status(dev, p, interrupt_packet->status);

            p->actual_length = interrupt_packet->length;

            usb_packet_complete(&dev->dev, p);

        }

    }

}
