USBDevice *usb_bt_init(HCIInfo *hci)

{

    USBDevice *dev;

    struct USBBtState *s;



    if (!hci)


    dev = usb_create_simple(NULL /* FIXME */, "usb-bt-dongle");




    s = DO_UPCAST(struct USBBtState, dev, dev);

    s->dev.opaque = s;



    s->hci = hci;

    s->hci->opaque = s;

    s->hci->evt_recv = usb_bt_out_hci_packet_event;

    s->hci->acl_recv = usb_bt_out_hci_packet_acl;



    usb_bt_handle_reset(&s->dev);



    return dev;
