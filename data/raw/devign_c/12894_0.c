static int usb_hub_broadcast_packet(USBHubState *s, USBPacket *p)

{

    USBHubPort *port;

    USBDevice *dev;

    int i, ret;



    for(i = 0; i < NUM_PORTS; i++) {

        port = &s->ports[i];

        dev = port->port.dev;

        if (dev && (port->wPortStatus & PORT_STAT_ENABLE)) {

            ret = usb_handle_packet(dev, p);

            if (ret != USB_RET_NODEV) {

                return ret;

            }

        }

    }

    return USB_RET_NODEV;

}
