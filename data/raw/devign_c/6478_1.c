static int usbredir_handle_bulk_data(USBRedirDevice *dev, USBPacket *p,

                                      uint8_t ep)

{

    AsyncURB *aurb = async_alloc(dev, p);

    struct usb_redir_bulk_packet_header bulk_packet;



    DPRINTF("bulk-out ep %02X len %d id %u\n", ep, p->len, aurb->packet_id);



    bulk_packet.endpoint  = ep;

    bulk_packet.length    = p->len;

    bulk_packet.stream_id = 0;

    aurb->bulk_packet = bulk_packet;



    if (ep & USB_DIR_IN) {

        usbredirparser_send_bulk_packet(dev->parser, aurb->packet_id,

                                        &bulk_packet, NULL, 0);

    } else {

        usbredir_log_data(dev, "bulk data out:", p->data, p->len);

        usbredirparser_send_bulk_packet(dev->parser, aurb->packet_id,

                                        &bulk_packet, p->data, p->len);

    }

    usbredirparser_do_write(dev->parser);

    return USB_RET_ASYNC;

}
