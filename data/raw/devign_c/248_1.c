static void usbredir_bulk_packet(void *priv, uint32_t id,

    struct usb_redir_bulk_packet_header *bulk_packet,

    uint8_t *data, int data_len)

{

    USBRedirDevice *dev = priv;

    uint8_t ep = bulk_packet->endpoint;

    int len = bulk_packet->length;

    AsyncURB *aurb;



    DPRINTF("bulk-in status %d ep %02X len %d id %u\n", bulk_packet->status,

            ep, len, id);



    aurb = async_find(dev, id);

    if (!aurb) {

        free(data);

        return;

    }



    if (aurb->bulk_packet.endpoint != bulk_packet->endpoint ||

            aurb->bulk_packet.stream_id != bulk_packet->stream_id) {

        ERROR("return bulk packet mismatch, please report this!\n");

        len = USB_RET_NAK;

    }



    if (aurb->packet) {

        len = usbredir_handle_status(dev, bulk_packet->status, len);

        if (len > 0) {

            usbredir_log_data(dev, "bulk data in:", data, data_len);

            if (data_len <= aurb->packet->len) {

                memcpy(aurb->packet->data, data, data_len);

            } else {

                ERROR("bulk buffer too small (%d > %d)\n", data_len,

                      aurb->packet->len);

                len = USB_RET_STALL;

            }

        }

        aurb->packet->len = len;

        usb_packet_complete(&dev->dev, aurb->packet);

    }

    async_free(dev, aurb);

    free(data);

}
