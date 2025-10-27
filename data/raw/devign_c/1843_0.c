static int usb_wacom_initfn(USBDevice *dev)

{

    USBWacomState *s = DO_UPCAST(USBWacomState, dev, dev);

    s->dev.speed = USB_SPEED_FULL;

    s->changed = 1;

    return 0;

}
