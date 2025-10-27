static void usbredir_do_attach(void *opaque)

{

    USBRedirDevice *dev = opaque;



    /* In order to work properly with XHCI controllers we need these caps */

    if ((dev->dev.port->speedmask & USB_SPEED_MASK_SUPER) && !(

        usbredirparser_peer_has_cap(dev->parser,

                                    usb_redir_cap_ep_info_max_packet_size) &&

        usbredirparser_peer_has_cap(dev->parser,

                                    usb_redir_cap_32bits_bulk_length) &&

        usbredirparser_peer_has_cap(dev->parser,

                                    usb_redir_cap_64bits_ids))) {

        ERROR("usb-redir-host lacks capabilities needed for use with XHCI\n");

        usbredir_reject_device(dev);

        return;

    }



    if (usb_device_attach(&dev->dev) != 0) {

        WARNING("rejecting device due to speed mismatch\n");

        usbredir_reject_device(dev);

    }

}
