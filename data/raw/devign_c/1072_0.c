static int usb_serial_initfn(USBDevice *dev)

{

    USBSerialState *s = DO_UPCAST(USBSerialState, dev, dev);

    s->dev.speed = USB_SPEED_FULL;



    if (!s->cs) {

        error_report("Property chardev is required");

        return -1;

    }



    qemu_chr_add_handlers(s->cs, usb_serial_can_read, usb_serial_read,

                          usb_serial_event, s);

    usb_serial_handle_reset(dev);

    return 0;

}
