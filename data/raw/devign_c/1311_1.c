static USBDevice *usb_braille_init(USBBus *bus, const char *unused)

{

    USBDevice *dev;

    CharDriverState *cdrv;



    cdrv = qemu_chr_new("braille", "braille", NULL);

    if (!cdrv)

        return NULL;



    dev = usb_create(bus, "usb-braille");

    qdev_prop_set_chr(&dev->qdev, "chardev", cdrv);

    qdev_init_nofail(&dev->qdev);



    return dev;

}
