static int usb_hub_handle_control(USBDevice *dev, int request, int value,

                                  int index, int length, uint8_t *data)

{

    USBHubState *s = (USBHubState *)dev;

    int ret;



    ret = usb_desc_handle_control(dev, request, value, index, length, data);

    if (ret >= 0) {

        return ret;

    }



    switch(request) {

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

    case EndpointOutRequest | USB_REQ_CLEAR_FEATURE:

        if (value == 0 && index != 0x81) { /* clear ep halt */

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

        /* usb specific requests */

    case GetHubStatus:

        data[0] = 0;

        data[1] = 0;

        data[2] = 0;

        data[3] = 0;

        ret = 4;

        break;

    case GetPortStatus:

        {

            unsigned int n = index - 1;

            USBHubPort *port;

            if (n >= NUM_PORTS) {

                goto fail;

            }

            port = &s->ports[n];

            data[0] = port->wPortStatus;

            data[1] = port->wPortStatus >> 8;

            data[2] = port->wPortChange;

            data[3] = port->wPortChange >> 8;

            ret = 4;

        }

        break;

    case SetHubFeature:

    case ClearHubFeature:

        if (value == 0 || value == 1) {

        } else {

            goto fail;

        }

        ret = 0;

        break;

    case SetPortFeature:

        {

            unsigned int n = index - 1;

            USBHubPort *port;

            USBDevice *dev;

            if (n >= NUM_PORTS) {

                goto fail;

            }

            port = &s->ports[n];

            dev = port->port.dev;

            switch(value) {

            case PORT_SUSPEND:

                port->wPortStatus |= PORT_STAT_SUSPEND;

                break;

            case PORT_RESET:

                if (dev) {

                    usb_send_msg(dev, USB_MSG_RESET);

                    port->wPortChange |= PORT_STAT_C_RESET;

                    /* set enable bit */

                    port->wPortStatus |= PORT_STAT_ENABLE;

                }

                break;

            case PORT_POWER:

                break;

            default:

                goto fail;

            }

            ret = 0;

        }

        break;

    case ClearPortFeature:

        {

            unsigned int n = index - 1;

            USBHubPort *port;



            if (n >= NUM_PORTS) {

                goto fail;

            }

            port = &s->ports[n];

            switch(value) {

            case PORT_ENABLE:

                port->wPortStatus &= ~PORT_STAT_ENABLE;

                break;

            case PORT_C_ENABLE:

                port->wPortChange &= ~PORT_STAT_C_ENABLE;

                break;

            case PORT_SUSPEND:

                port->wPortStatus &= ~PORT_STAT_SUSPEND;

                break;

            case PORT_C_SUSPEND:

                port->wPortChange &= ~PORT_STAT_C_SUSPEND;

                break;

            case PORT_C_CONNECTION:

                port->wPortChange &= ~PORT_STAT_C_CONNECTION;

                break;

            case PORT_C_OVERCURRENT:

                port->wPortChange &= ~PORT_STAT_C_OVERCURRENT;

                break;

            case PORT_C_RESET:

                port->wPortChange &= ~PORT_STAT_C_RESET;

                break;

            default:

                goto fail;

            }

            ret = 0;

        }

        break;

    case GetHubDescriptor:

        {

            unsigned int n, limit, var_hub_size = 0;

            memcpy(data, qemu_hub_hub_descriptor,

                   sizeof(qemu_hub_hub_descriptor));

            data[2] = NUM_PORTS;



            /* fill DeviceRemovable bits */

            limit = ((NUM_PORTS + 1 + 7) / 8) + 7;

            for (n = 7; n < limit; n++) {

                data[n] = 0x00;

                var_hub_size++;

            }



            /* fill PortPwrCtrlMask bits */

            limit = limit + ((NUM_PORTS + 7) / 8);

            for (;n < limit; n++) {

                data[n] = 0xff;

                var_hub_size++;

            }



            ret = sizeof(qemu_hub_hub_descriptor) + var_hub_size;

            data[0] = ret;

            break;

        }

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
