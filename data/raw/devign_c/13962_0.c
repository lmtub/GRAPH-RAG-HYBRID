static void usbredir_control_packet(void *priv, uint32_t id,

    struct usb_redir_control_packet_header *control_packet,

    uint8_t *data, int data_len)

{

    USBRedirDevice *dev = priv;

    int len = control_packet->length;

    AsyncURB *aurb;



    DPRINTF("ctrl-in status %d len %d id %u\n", control_packet->status,

            len, id);



    aurb = async_find(dev, id);

    if (!aurb) {

        free(data);

        return;

    }



    aurb->control_packet.status = control_packet->status;

    aurb->control_packet.length = control_packet->length;

    if (memcmp(&aurb->control_packet, control_packet,

               sizeof(*control_packet))) {

        ERROR("return control packet mismatch, please report this!\n");

        len = USB_RET_NAK;

    }



    if (aurb->packet) {

        len = usbredir_handle_status(dev, control_packet->status, len);

        if (len > 0) {

            usbredir_log_data(dev, "ctrl data in:", data, data_len);

            if (data_len <= sizeof(dev->dev.data_buf)) {

                memcpy(dev->dev.data_buf, data, data_len);

            } else {

                ERROR("ctrl buffer too small (%d > %zu)\n",

                      data_len, sizeof(dev->dev.data_buf));

                len = USB_RET_STALL;

            }

        }

        aurb->packet->result = len;

        usb_generic_async_ctrl_complete(&dev->dev, aurb->packet);

    }

    async_free(dev, aurb);

    free(data);

}
