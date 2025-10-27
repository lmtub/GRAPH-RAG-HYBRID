static void usb_hid_handle_destroy(USBDevice *dev)

{

    USBHIDState *s = (USBHIDState *)dev;



    if (s->kind != USB_KEYBOARD)

        qemu_remove_mouse_event_handler(s->ptr.eh_entry);

    /* TODO: else */

}
