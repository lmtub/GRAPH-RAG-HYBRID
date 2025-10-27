static int usb_wacom_handle_data(USBDevice *dev, USBPacket *p)

{

    USBWacomState *s = (USBWacomState *) dev;

    uint8_t buf[p->iov.size];

    int ret = 0;



    switch (p->pid) {

    case USB_TOKEN_IN:

        if (p->devep == 1) {

            if (!(s->changed || s->idle))

                return USB_RET_NAK;

            s->changed = 0;

            if (s->mode == WACOM_MODE_HID)

                ret = usb_mouse_poll(s, buf, p->iov.size);

            else if (s->mode == WACOM_MODE_WACOM)

                ret = usb_wacom_poll(s, buf, p->iov.size);

            usb_packet_copy(p, buf, ret);

            break;

        }

        /* Fall through.  */

    case USB_TOKEN_OUT:

    default:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
