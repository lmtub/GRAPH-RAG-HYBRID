static void do_token_in(USBDevice *s, USBPacket *p)

{

    int request, value, index;



    assert(p->ep->nr == 0);



    request = (s->setup_buf[0] << 8) | s->setup_buf[1];

    value   = (s->setup_buf[3] << 8) | s->setup_buf[2];

    index   = (s->setup_buf[5] << 8) | s->setup_buf[4];

 

    switch(s->setup_state) {

    case SETUP_STATE_ACK:

        if (!(s->setup_buf[0] & USB_DIR_IN)) {

            usb_device_handle_control(s, p, request, value, index,

                                      s->setup_len, s->data_buf);

            if (p->status == USB_RET_ASYNC) {

                return;

            }

            s->setup_state = SETUP_STATE_IDLE;

            p->actual_length = 0;

        }

        break;



    case SETUP_STATE_DATA:

        if (s->setup_buf[0] & USB_DIR_IN) {

            int len = s->setup_len - s->setup_index;

            if (len > p->iov.size) {

                len = p->iov.size;

            }

            usb_packet_copy(p, s->data_buf + s->setup_index, len);

            s->setup_index += len;

            if (s->setup_index >= s->setup_len) {

                s->setup_state = SETUP_STATE_ACK;

            }

            return;

        }

        s->setup_state = SETUP_STATE_IDLE;

        p->status = USB_RET_STALL;

        break;



    default:

        p->status = USB_RET_STALL;

    }

}
