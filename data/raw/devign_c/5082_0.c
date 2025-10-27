static int usb_qdev_exit(DeviceState *qdev)

{

    USBDevice *dev = USB_DEVICE(qdev);



    if (dev->attached) {

        usb_device_detach(dev);

    }

    usb_device_handle_destroy(dev);

    if (dev->port) {

        usb_release_port(dev);

    }

    return 0;

}
