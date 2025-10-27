USBDevice *usb_net_init(NICInfo *nd)

{

    USBNetState *s;



    s = qemu_mallocz(sizeof(USBNetState));

    s->dev.speed = USB_SPEED_FULL;

    s->dev.handle_packet = usb_generic_handle_packet;



    s->dev.handle_reset = usb_net_handle_reset;

    s->dev.handle_control = usb_net_handle_control;

    s->dev.handle_data = usb_net_handle_data;

    s->dev.handle_destroy = usb_net_handle_destroy;



    s->rndis = 1;

    s->rndis_state = RNDIS_UNINITIALIZED;

    s->medium = 0;	/* NDIS_MEDIUM_802_3 */

    s->speed = 1000000; /* 100MBps, in 100Bps units */

    s->media_state = 0;	/* NDIS_MEDIA_STATE_CONNECTED */;

    s->filter = 0;

    s->vendorid = 0x1234;



    memcpy(s->mac, nd->macaddr, 6);

    TAILQ_INIT(&s->rndis_resp);



    pstrcpy(s->dev.devname, sizeof(s->dev.devname),

                    "QEMU USB Network Interface");

    s->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                    usbnet_receive, usbnet_can_receive, s);



    qemu_format_nic_info_str(s->vc, s->mac);



    snprintf(s->usbstring_mac, sizeof(s->usbstring_mac),

                    "%02x%02x%02x%02x%02x%02x",

                    0x40, s->mac[1], s->mac[2],

                    s->mac[3], s->mac[4], s->mac[5]);

    fprintf(stderr, "usbnet: initialized mac %02x:%02x:%02x:%02x:%02x:%02x\n",

                    s->mac[0], s->mac[1], s->mac[2],

                    s->mac[3], s->mac[4], s->mac[5]);



    return (USBDevice *) s;

}
