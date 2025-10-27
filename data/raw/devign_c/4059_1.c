USBPacket *usb_ep_find_packet_by_id(USBDevice *dev, int pid, int ep,

                                    uint64_t id)

{

    struct USBEndpoint *uep = usb_ep_get(dev, pid, ep);

    USBPacket *p;



    while ((p = QTAILQ_FIRST(&uep->queue)) != NULL) {

        if (p->id == id) {

            return p;

        }

    }



    return NULL;

}
