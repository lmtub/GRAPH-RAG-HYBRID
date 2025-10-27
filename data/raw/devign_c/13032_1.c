void usb_packet_setup(USBPacket *p, int pid, USBEndpoint *ep, uint64_t id,

                      bool short_not_ok, bool int_req)

{

    assert(!usb_packet_is_inflight(p));

    assert(p->iov.iov != NULL);

    p->id = id;

    p->pid = pid;

    p->ep = ep;

    p->result = 0;

    p->parameter = 0;

    p->short_not_ok = short_not_ok;

    p->int_req = int_req;


    qemu_iovec_reset(&p->iov);

    usb_packet_set_state(p, USB_PACKET_SETUP);

}