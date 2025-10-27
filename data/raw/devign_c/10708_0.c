static int usb_bt_initfn(USBDevice *dev)

{

    struct USBBtState *s = DO_UPCAST(struct USBBtState, dev, dev);

    s->dev.speed = USB_SPEED_HIGH;

    return 0;

}
