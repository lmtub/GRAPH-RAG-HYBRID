void usb_cancel_packet(USBPacket * p)

{

    assert(p->owner != NULL);

    usb_device_cancel_packet(p->owner->dev, p);

    p->owner = NULL;

}
