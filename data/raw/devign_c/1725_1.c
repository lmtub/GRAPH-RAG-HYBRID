static int ehci_execute(EHCIPacket *p, const char *action)

{

    USBEndpoint *ep;

    int ret;

    int endp;



    if (!(p->qtd.token & QTD_TOKEN_ACTIVE)) {

        fprintf(stderr, "Attempting to execute inactive qtd\n");

        return USB_RET_PROCERR;

    }



    p->tbytes = (p->qtd.token & QTD_TOKEN_TBYTES_MASK) >> QTD_TOKEN_TBYTES_SH;

    if (p->tbytes > BUFF_SIZE) {

        ehci_trace_guest_bug(p->queue->ehci,

                             "guest requested more bytes than allowed");

        return USB_RET_PROCERR;

    }



    p->pid = (p->qtd.token & QTD_TOKEN_PID_MASK) >> QTD_TOKEN_PID_SH;

    switch (p->pid) {

    case 0:

        p->pid = USB_TOKEN_OUT;

        break;

    case 1:

        p->pid = USB_TOKEN_IN;

        break;

    case 2:

        p->pid = USB_TOKEN_SETUP;

        break;

    default:

        fprintf(stderr, "bad token\n");

        break;

    }



    if (ehci_init_transfer(p) != 0) {

        return USB_RET_PROCERR;

    }



    endp = get_field(p->queue->qh.epchar, QH_EPCHAR_EP);

    ep = usb_ep_get(p->queue->dev, p->pid, endp);



    usb_packet_setup(&p->packet, p->pid, ep, p->qtdaddr);

    usb_packet_map(&p->packet, &p->sgl);



    trace_usb_ehci_packet_action(p->queue, p, action);

    ret = usb_handle_packet(p->queue->dev, &p->packet);

    DPRINTF("submit: qh %x next %x qtd %x pid %x len %zd "

            "(total %d) endp %x ret %d\n",

            q->qhaddr, q->qh.next, q->qtdaddr, q->pid,

            q->packet.iov.size, q->tbytes, endp, ret);



    if (ret > BUFF_SIZE) {

        fprintf(stderr, "ret from usb_handle_packet > BUFF_SIZE\n");

        return USB_RET_PROCERR;

    }



    return ret;

}
