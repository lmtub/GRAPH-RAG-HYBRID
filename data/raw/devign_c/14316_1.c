static void usb_serial_handle_data(USBDevice *dev, USBPacket *p)

{

    USBSerialState *s = (USBSerialState *)dev;

    uint8_t devep = p->ep->nr;

    struct iovec *iov;

    uint8_t header[2];

    int i, first_len, len;



    switch (p->pid) {

    case USB_TOKEN_OUT:

        if (devep != 2)

            goto fail;

        for (i = 0; i < p->iov.niov; i++) {

            iov = p->iov.iov + i;

            qemu_chr_fe_write(s->cs, iov->iov_base, iov->iov_len);

        }

        p->actual_length = p->iov.size;

        break;



    case USB_TOKEN_IN:

        if (devep != 1)

            goto fail;

        first_len = RECV_BUF - s->recv_ptr;

        len = p->iov.size;

        if (len <= 2) {

            p->status = USB_RET_NAK;

            break;

        }

        header[0] = usb_get_modem_lines(s) | 1;

        /* We do not have the uart details */

        /* handle serial break */

        if (s->event_trigger && s->event_trigger & FTDI_BI) {

            s->event_trigger &= ~FTDI_BI;

            header[1] = FTDI_BI;

            usb_packet_copy(p, header, 2);

            break;

        } else {

            header[1] = 0;

        }

        len -= 2;

        if (len > s->recv_used)

            len = s->recv_used;

        if (!len) {

            p->status = USB_RET_NAK;

            break;

        }

        if (first_len > len)

            first_len = len;

        usb_packet_copy(p, header, 2);

        usb_packet_copy(p, s->recv_buf + s->recv_ptr, first_len);

        if (len > first_len)

            usb_packet_copy(p, s->recv_buf, len - first_len);

        s->recv_used -= len;

        s->recv_ptr = (s->recv_ptr + len) % RECV_BUF;

        break;



    default:

        DPRINTF("Bad token\n");

    fail:

        p->status = USB_RET_STALL;

        break;

    }

}
