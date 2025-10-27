int usb_desc_handle_control(USBDevice *dev, USBPacket *p,

        int request, int value, int index, int length, uint8_t *data)

{

    const USBDesc *desc = usb_device_get_usb_desc(dev);

    int ret = -1;



    assert(desc != NULL);

    switch(request) {

    case DeviceOutRequest | USB_REQ_SET_ADDRESS:

        dev->addr = value;

        trace_usb_set_addr(dev->addr);

        ret = 0;

        break;



    case DeviceRequest | USB_REQ_GET_DESCRIPTOR:

        ret = usb_desc_get_descriptor(dev, value, data, length);

        break;



    case DeviceRequest | USB_REQ_GET_CONFIGURATION:

        data[0] = dev->config->bConfigurationValue;

        ret = 1;

        break;

    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

        ret = usb_desc_set_config(dev, value);

        trace_usb_set_config(dev->addr, value, ret);

        break;



    case DeviceRequest | USB_REQ_GET_STATUS:

        data[0] = 0;

        if (dev->config->bmAttributes & 0x40) {

            data[0] |= 1 << USB_DEVICE_SELF_POWERED;

        }

        if (dev->remote_wakeup) {

            data[0] |= 1 << USB_DEVICE_REMOTE_WAKEUP;

        }

        data[1] = 0x00;

        ret = 2;

        break;

    case DeviceOutRequest | USB_REQ_CLEAR_FEATURE:

        if (value == USB_DEVICE_REMOTE_WAKEUP) {

            dev->remote_wakeup = 0;

            ret = 0;

        }

        trace_usb_clear_device_feature(dev->addr, value, ret);

        break;

    case DeviceOutRequest | USB_REQ_SET_FEATURE:

        if (value == USB_DEVICE_REMOTE_WAKEUP) {

            dev->remote_wakeup = 1;

            ret = 0;

        }

        trace_usb_set_device_feature(dev->addr, value, ret);

        break;



    case InterfaceRequest | USB_REQ_GET_INTERFACE:

        if (index < 0 || index >= dev->ninterfaces) {

            break;

        }

        data[0] = dev->altsetting[index];

        ret = 1;

        break;

    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        ret = usb_desc_set_interface(dev, index, value);

        trace_usb_set_interface(dev->addr, index, value, ret);

        break;



    }

    return ret;

}
