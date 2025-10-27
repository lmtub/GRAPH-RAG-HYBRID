static void softusb_usbdev_datain(void *opaque)

{

    MilkymistSoftUsbState *s = opaque;



    USBPacket p;



    p.pid = USB_TOKEN_IN;

    p.devep = 1;

    p.data = s->kbd_usb_buffer;

    p.len = sizeof(s->kbd_usb_buffer);

    s->usbdev->info->handle_data(s->usbdev, &p);



    softusb_kbd_changed(s);

}
