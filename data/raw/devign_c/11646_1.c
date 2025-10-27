static void ehci_free_packet(EHCIPacket *p)

{

    trace_usb_ehci_packet_action(p->queue, p, "free");

    if (p->async == EHCI_ASYNC_INFLIGHT) {

        usb_cancel_packet(&p->packet);

        usb_packet_unmap(&p->packet, &p->sgl);

        qemu_sglist_destroy(&p->sgl);












    QTAILQ_REMOVE(&p->queue->packets, p, next);

    usb_packet_cleanup(&p->packet);

    g_free(p);
