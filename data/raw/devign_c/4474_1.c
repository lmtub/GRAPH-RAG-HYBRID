static int usb_net_handle_datain(USBNetState *s, USBPacket *p)

{

    int ret = USB_RET_NAK;



    if (s->in_ptr > s->in_len) {

        s->in_ptr = s->in_len = 0;

        ret = USB_RET_NAK;

        return ret;

    }

    if (!s->in_len) {

        ret = USB_RET_NAK;

        return ret;

    }

    ret = s->in_len - s->in_ptr;

    if (ret > p->len)

        ret = p->len;

    memcpy(p->data, &s->in_buf[s->in_ptr], ret);

    s->in_ptr += ret;

    if (s->in_ptr >= s->in_len &&

                    (is_rndis(s) || (s->in_len & (64 - 1)) || !ret)) {

        /* no short packet necessary */

        s->in_ptr = s->in_len = 0;

    }



#ifdef TRAFFIC_DEBUG

    fprintf(stderr, "usbnet: data in len %u return %d", p->len, ret);

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
