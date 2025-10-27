static void ehci_async_complete_packet(USBPort *port, USBPacket *packet)

{

    EHCIQueue *q;

    EHCIState *s = port->opaque;

    uint32_t portsc = s->portsc[port->index];



    if (portsc & PORTSC_POWNER) {

        USBPort *companion = s->companion_ports[port->index];

        companion->ops->complete(companion, packet);

        return;

    }



    q = container_of(packet, EHCIQueue, packet);

    trace_usb_ehci_queue_action(q, "wakeup");

    assert(q->async == EHCI_ASYNC_INFLIGHT);

    q->async = EHCI_ASYNC_FINISHED;

    q->usb_status = packet->len;

}
