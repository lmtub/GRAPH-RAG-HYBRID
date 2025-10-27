static void usb_hub_handle_attach(USBDevice *dev)

{

    USBHubState *s = DO_UPCAST(USBHubState, dev, dev);

    int i;



    for (i = 0; i < NUM_PORTS; i++) {

        usb_port_location(&s->ports[i].port, dev->port, i+1);

    }

}
