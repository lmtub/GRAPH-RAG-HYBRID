static int usb_hid_initfn(USBDevice *dev, int kind)

{

    USBHIDState *us = DO_UPCAST(USBHIDState, dev, dev);



    usb_desc_init(dev);

    hid_init(&us->hid, kind, usb_hid_changed);



    /* Force poll routine to be run and grab input the first time.  */

    us->changed = 1;

    return 0;

}
