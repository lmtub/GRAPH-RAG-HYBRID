static int usb_net_handle_control(USBDevice *dev, int request, int value,

                int index, int length, uint8_t *data)

{

    USBNetState *s = (USBNetState *) dev;

    int ret = 0;



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



    case DeviceOutRequest | USB_REQ_SET_FEATURE:

        if (value == USB_DEVICE_REMOTE_WAKEUP) {

            dev->remote_wakeup = 1;

        } else {

            goto fail;

        }

        ret = 0;

        break;



    case DeviceOutRequest | USB_REQ_SET_ADDRESS:

        dev->addr = value;

        ret = 0;

        break;



    case ClassInterfaceOutRequest | USB_CDC_SEND_ENCAPSULATED_COMMAND:

        if (!s->rndis || value || index != 0)

            goto fail;

#ifdef TRAFFIC_DEBUG

        {

            unsigned int i;

            fprintf(stderr, "SEND_ENCAPSULATED_COMMAND:");

            for (i = 0; i < length; i++) {

                if (!(i & 15))

                    fprintf(stderr, "\n%04x:", i);

                fprintf(stderr, " %02x", data[i]);

            }

            fprintf(stderr, "\n\n");

        }

#endif

        ret = rndis_parse(s, data, length);

        break;



    case ClassInterfaceRequest | USB_CDC_GET_ENCAPSULATED_RESPONSE:

        if (!s->rndis || value || index != 0)

            goto fail;

        ret = rndis_get_response(s, data);

        if (!ret) {

            data[0] = 0;

            ret = 1;

        }

#ifdef TRAFFIC_DEBUG

        {

            unsigned int i;

            fprintf(stderr, "GET_ENCAPSULATED_RESPONSE:");

            for (i = 0; i < ret; i++) {

                if (!(i & 15))

                    fprintf(stderr, "\n%04x:", i);

                fprintf(stderr, " %02x", data[i]);

            }

            fprintf(stderr, "\n\n");

        }

#endif

        break;



    case DeviceRequest | USB_REQ_GET_DESCRIPTOR:

        switch(value >> 8) {

        case USB_DT_DEVICE:

            ret = sizeof(qemu_net_dev_descriptor);

            memcpy(data, qemu_net_dev_descriptor, ret);

            break;



        case USB_DT_CONFIG:

            switch (value & 0xff) {

            case 0:

                ret = sizeof(qemu_net_rndis_config_descriptor);

                memcpy(data, qemu_net_rndis_config_descriptor, ret);

                break;



            case 1:

                ret = sizeof(qemu_net_cdc_config_descriptor);

                memcpy(data, qemu_net_cdc_config_descriptor, ret);

                break;



            default:

                goto fail;

            }



            data[2] = ret & 0xff;

            data[3] = ret >> 8;

            break;



        case USB_DT_STRING:

            switch (value & 0xff) {

            case 0:

                /* language ids */

                data[0] = 4;

                data[1] = 3;

                data[2] = 0x09;

                data[3] = 0x04;

                ret = 4;

                break;



            case STRING_ETHADDR:

                ret = set_usb_string(data, s->usbstring_mac);

                break;



            default:

                if (usb_net_stringtable[value & 0xff]) {

                    ret = set_usb_string(data,

                                    usb_net_stringtable[value & 0xff]);

                    break;

                }



                goto fail;

            }

            break;



        default:

            goto fail;

        }

        break;



    case DeviceRequest | USB_REQ_GET_CONFIGURATION:

        data[0] = s->rndis ? DEV_RNDIS_CONFIG_VALUE : DEV_CONFIG_VALUE;

        ret = 1;

        break;



    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

        switch (value & 0xff) {

        case DEV_CONFIG_VALUE:

            s->rndis = 0;

            break;



        case DEV_RNDIS_CONFIG_VALUE:

            s->rndis = 1;

            break;



        default:

            goto fail;

        }

        ret = 0;

        break;



    case DeviceRequest | USB_REQ_GET_INTERFACE:

    case InterfaceRequest | USB_REQ_GET_INTERFACE:

        data[0] = 0;

        ret = 1;

        break;



    case DeviceOutRequest | USB_REQ_SET_INTERFACE:

    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        ret = 0;

        break;



    default:

    fail:

        fprintf(stderr, "usbnet: failed control transaction: "

                        "request 0x%x value 0x%x index 0x%x length 0x%x\n",

                        request, value, index, length);

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
