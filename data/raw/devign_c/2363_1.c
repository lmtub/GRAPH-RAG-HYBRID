static USBDevice *usb_bt_init(USBBus *bus, const char *cmdline)

{

    USBDevice *dev;

    struct USBBtState *s;

    HCIInfo *hci;

    const char *name = "usb-bt-dongle";



    if (*cmdline) {

        hci = hci_init(cmdline);

    } else {

        hci = bt_new_hci(qemu_find_bt_vlan(0));

    }

    if (!hci)

        return NULL;



    dev = usb_create(bus, name);

    s = DO_UPCAST(struct USBBtState, dev, dev);

    s->hci = hci;

    if (qdev_init(&dev->qdev) < 0) {

        error_report("Failed to initialize USB device '%s'", name);

        return NULL;

    }



    return dev;

}
