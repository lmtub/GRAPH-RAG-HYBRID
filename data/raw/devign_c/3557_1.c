static int usb_serial_handle_data(USBDevice *dev, USBPacket *p)

{

    USBSerialState *s = (USBSerialState *)dev;

    int ret = 0;

    uint8_t devep = p->devep;

    uint8_t *data = p->data;

    int len = p->len;

    int first_len;



    switch (p->pid) {

    case USB_TOKEN_OUT:

        if (devep != 2)

            goto fail;

        qemu_chr_write(s->cs, data, len);

        break;



    case USB_TOKEN_IN:

        if (devep != 1)

            goto fail;

        first_len = RECV_BUF - s->recv_ptr;

        if (len <= 2) {

            ret = USB_RET_NAK;

            break;

        }

        *data++ = usb_get_modem_lines(s) | 1;

        /* We do not have the uart details */

        /* handle serial break */

        if (s->event_trigger && s->event_trigger & FTDI_BI) {

            s->event_trigger &= ~FTDI_BI;

            *data = FTDI_BI;

            ret = 2;

            break;

        } else {

            *data++ = 0;

        }

        len -= 2;

        if (len > s->recv_used)

            len = s->recv_used;

        if (!len) {

            ret = USB_RET_NAK;

            break;

        }

        if (first_len > len)

            first_len = len;

        memcpy(data, s->recv_buf + s->recv_ptr, first_len);

        if (len > first_len)

            memcpy(data + first_len, s->recv_buf, len - first_len);

        s->recv_used -= len;

        s->recv_ptr = (s->recv_ptr + len) % RECV_BUF;

        ret = len + 2;

        break;



    default:

        DPRINTF("Bad token\n");

    fail:

        ret = USB_RET_STALL;

        break;

    }



    return ret;

}
