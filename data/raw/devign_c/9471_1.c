static int ehci_execute(EHCIQueue *q)

{

    USBPort *port;

    USBDevice *dev;

    int ret;

    int i;

    int endp;

    int devadr;



    if ( !(q->qh.token & QTD_TOKEN_ACTIVE)) {

        fprintf(stderr, "Attempting to execute inactive QH\n");

        return USB_RET_PROCERR;

    }



    q->tbytes = (q->qh.token & QTD_TOKEN_TBYTES_MASK) >> QTD_TOKEN_TBYTES_SH;

    if (q->tbytes > BUFF_SIZE) {

        fprintf(stderr, "Request for more bytes than allowed\n");

        return USB_RET_PROCERR;

    }



    q->pid = (q->qh.token & QTD_TOKEN_PID_MASK) >> QTD_TOKEN_PID_SH;

    switch(q->pid) {

        case 0: q->pid = USB_TOKEN_OUT; break;

        case 1: q->pid = USB_TOKEN_IN; break;

        case 2: q->pid = USB_TOKEN_SETUP; break;

        default: fprintf(stderr, "bad token\n"); break;

    }



    if ((q->tbytes && q->pid != USB_TOKEN_IN) &&

        (ehci_buffer_rw(q, q->tbytes, 0) != 0)) {

        return USB_RET_PROCERR;

    }



    endp = get_field(q->qh.epchar, QH_EPCHAR_EP);

    devadr = get_field(q->qh.epchar, QH_EPCHAR_DEVADDR);



    ret = USB_RET_NODEV;



    // TO-DO: associating device with ehci port

    for(i = 0; i < NB_PORTS; i++) {

        port = &q->ehci->ports[i];

        dev = port->dev;



        if (!(q->ehci->portsc[i] &(PORTSC_CONNECT))) {

            DPRINTF("Port %d, no exec, not connected(%08X)\n",

                    i, q->ehci->portsc[i]);

            continue;

        }



        q->packet.pid = q->pid;

        q->packet.devaddr = devadr;

        q->packet.devep = endp;

        q->packet.data = q->buffer;

        q->packet.len = q->tbytes;



        ret = usb_handle_packet(dev, &q->packet);



        DPRINTF("submit: qh %x next %x qtd %x pid %x len %d (total %d) endp %x ret %d\n",

                q->qhaddr, q->qh.next, q->qtdaddr, q->pid,

                q->packet.len, q->tbytes, endp, ret);



        if (ret != USB_RET_NODEV) {

            break;

        }

    }



    if (ret > BUFF_SIZE) {

        fprintf(stderr, "ret from usb_handle_packet > BUFF_SIZE\n");

        return USB_RET_PROCERR;

    }



    return ret;

}
