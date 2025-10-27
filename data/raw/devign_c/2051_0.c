static USBDevice *usb_serial_init(USBBus *bus, const char *filename)

{

    USBDevice *dev;

    Chardev *cdrv;

    char label[32];

    static int index;



    while (*filename && *filename != ':') {

        const char *p;



        if (strstart(filename, "vendorid=", &p)) {

            error_report("vendorid is not supported anymore");

            return NULL;

        } else if (strstart(filename, "productid=", &p)) {

            error_report("productid is not supported anymore");

            return NULL;

        } else {

            error_report("unrecognized serial USB option %s", filename);

            return NULL;

        }

        while(*filename == ',')

            filename++;

    }

    if (!*filename) {

        error_report("character device specification needed");

        return NULL;

    }

    filename++;



    snprintf(label, sizeof(label), "usbserial%d", index++);

    cdrv = qemu_chr_new(label, filename);

    if (!cdrv)

        return NULL;



    dev = usb_create(bus, "usb-serial");

    qdev_prop_set_chr(&dev->qdev, "chardev", cdrv);



    return dev;

}
