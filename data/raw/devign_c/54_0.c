static void usb_hid_changed(HIDState *hs)

{

    USBHIDState *us = container_of(hs, USBHIDState, hid);



    us->changed = 1;



    if (us->datain) {

        us->datain(us->datain_opaque);

    }



    usb_wakeup(&us->dev);

}
