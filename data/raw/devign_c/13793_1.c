static int usb_wacom_handle_data(USBDevice *dev, USBPacket *p)

{

    USBWacomState *s = (USBWacomState *) dev;

    int ret = 0;



    switch (p->pid) {

    case USB_TOKEN_IN:

        if (p->devep == 1) {

            if (!(s->changed || s->idle))

                return USB_RET_NAK;

            s->changed = 0;

            if (s->mode == WACOM_MODE_HID)

                ret = usb_mouse_poll(s, p->data, p->len);

            else if (s->mode == WACOM_MODE_WACOM)

                ret = usb_wacom_poll(s, p->data, p->len);

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
