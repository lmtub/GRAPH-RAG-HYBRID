static int usb_wacom_handle_control(USBDevice *dev, int request, int value,

                                    int index, int length, uint8_t *data)

{

    USBWacomState *s = (USBWacomState *) dev;

    int ret;



    ret = usb_desc_handle_control(dev, request, value, index, length, data);

    if (ret >= 0) {

        return ret;

    }



    ret = 0;

    switch (request) {

    case DeviceRequest | USB_REQ_GET_STATUS:

        data[0] = (1 << USB_DEVICE_SELF_POWERED) |

            (dev->remote_wakeup << USB_DEVICE_REMOTE_WAKEUP);

        data[1] = 0x00;

        ret = 2;

        break;

    case DeviceOutRequest | USB_REQ_CLEAR_FEATURE:

        if (value == USB_DEVICE_REMOTE_WAKEUP) {

            dev->remote_wakeup = 0;

        } else {

            goto fail;

        }

        ret = 0;

        break;

    case DeviceOutRequest | USB_REQ_SET_FEATURE:

        if (value == USB_DEVICE_REMOTE_WAKEUP) {

            dev->remote_wakeup = 1;

        } else {

            goto fail;

        }

        ret = 0;

        break;

    case DeviceRequest | USB_REQ_GET_CONFIGURATION:

        data[0] = 1;

        ret = 1;

        break;

    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

        ret = 0;

        break;

    case DeviceRequest | USB_REQ_GET_INTERFACE:

        data[0] = 0;

        ret = 1;

        break;

    case DeviceOutRequest | USB_REQ_SET_INTERFACE:

        ret = 0;

        break;

    case WACOM_SET_REPORT:

        if (s->mouse_grabbed) {

            qemu_remove_mouse_event_handler(s->eh_entry);

            s->mouse_grabbed = 0;

        }

        s->mode = data[0];

        ret = 0;

        break;

    case WACOM_GET_REPORT:

        data[0] = 0;

        data[1] = s->mode;

        ret = 2;

        break;

    /* USB HID requests */

    case HID_GET_REPORT:

        if (s->mode == WACOM_MODE_HID)

            ret = usb_mouse_poll(s, data, length);

        else if (s->mode == WACOM_MODE_WACOM)

            ret = usb_wacom_poll(s, data, length);

        break;

    case HID_GET_IDLE:

        ret = 1;

        data[0] = s->idle;

        break;

    case HID_SET_IDLE:

        s->idle = (uint8_t) (value >> 8);

        ret = 0;

        break;

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
