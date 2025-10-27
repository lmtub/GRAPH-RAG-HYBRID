static int usb_net_handle_statusin(USBNetState *s, USBPacket *p)

{

    int ret = 8;



    if (p->len < 8)

        return USB_RET_STALL;



    ((le32 *) p->data)[0] = cpu_to_le32(1);

    ((le32 *) p->data)[1] = cpu_to_le32(0);

    if (!s->rndis_resp.tqh_first)

        ret = USB_RET_NAK;



#ifdef TRAFFIC_DEBUG

    fprintf(stderr, "usbnet: interrupt poll len %u return %d", p->len, ret);

    {

        int i;

        fprintf(stderr, ":");

        for (i = 0; i < ret; i++) {

            if (!(i & 15))

                fprintf(stderr, "\n%04x:", i);

            fprintf(stderr, " %02x", p->data[i]);

        }

        fprintf(stderr, "\n\n");

    }

#endif



    return ret;

}
