static void usbredir_configuration_status(void *priv, uint32_t id,

    struct usb_redir_configuration_status_header *config_status)

{

    USBRedirDevice *dev = priv;

    AsyncURB *aurb;

    int len = 0;



    DPRINTF("set config status %d config %d id %u\n", config_status->status,

            config_status->configuration, id);



    aurb = async_find(dev, id);

    if (!aurb) {

        return;

    }

    if (aurb->packet) {

        if (aurb->get) {

            dev->dev.data_buf[0] = config_status->configuration;

            len = 1;

        }

        aurb->packet->len =

            usbredir_handle_status(dev, config_status->status, len);

        usb_generic_async_ctrl_complete(&dev->dev, aurb->packet);

    }

    async_free(dev, aurb);

}
