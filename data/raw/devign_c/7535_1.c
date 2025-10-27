static void usbredir_alt_setting_status(void *priv, uint32_t id,

    struct usb_redir_alt_setting_status_header *alt_setting_status)

{

    USBRedirDevice *dev = priv;

    AsyncURB *aurb;

    int len = 0;



    DPRINTF("alt status %d intf %d alt %d id: %u\n",

            alt_setting_status->status,

            alt_setting_status->interface,

            alt_setting_status->alt, id);



    aurb = async_find(dev, id);

    if (!aurb) {

        return;

    }

    if (aurb->packet) {

        if (aurb->get) {

            dev->dev.data_buf[0] = alt_setting_status->alt;

            len = 1;

        }

        aurb->packet->len =

            usbredir_handle_status(dev, alt_setting_status->status, len);

        usb_generic_async_ctrl_complete(&dev->dev, aurb->packet);

    }

    async_free(dev, aurb);

}
