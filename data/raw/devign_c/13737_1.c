static void usbredir_interface_info(void *priv,

    struct usb_redir_interface_info_header *interface_info)

{

    USBRedirDevice *dev = priv;



    dev->interface_info = *interface_info;



    /*

     * If we receive interface info after the device has already been

     * connected (ie on a set_config), re-check the filter.

     */

    if (qemu_timer_pending(dev->attach_timer) || dev->dev.attached) {

        if (usbredir_check_filter(dev)) {

            ERROR("Device no longer matches filter after interface info "

                  "change, disconnecting!\n");

        }

    }

}
