static void ehci_execute_complete(EHCIQueue *q)

{

    EHCIPacket *p = QTAILQ_FIRST(&q->packets);



    assert(p != NULL);

    assert(p->qtdaddr == q->qtdaddr);

    assert(p->async != EHCI_ASYNC_INFLIGHT);

    p->async = EHCI_ASYNC_NONE;



    DPRINTF("execute_complete: qhaddr 0x%x, next %x, qtdaddr 0x%x, status %d\n",

            q->qhaddr, q->qh.next, q->qtdaddr, q->usb_status);



    if (p->usb_status < 0) {

        switch (p->usb_status) {

        case USB_RET_IOERROR:

        case USB_RET_NODEV:

            q->qh.token |= (QTD_TOKEN_HALT | QTD_TOKEN_XACTERR);

            set_field(&q->qh.token, 0, QTD_TOKEN_CERR);

            ehci_record_interrupt(q->ehci, USBSTS_ERRINT);

            break;

        case USB_RET_STALL:

            q->qh.token |= QTD_TOKEN_HALT;

            ehci_record_interrupt(q->ehci, USBSTS_ERRINT);

            break;

        case USB_RET_NAK:

            set_field(&q->qh.altnext_qtd, 0, QH_ALTNEXT_NAKCNT);

            return; /* We're not done yet with this transaction */

        case USB_RET_BABBLE:

            q->qh.token |= (QTD_TOKEN_HALT | QTD_TOKEN_BABBLE);

            ehci_record_interrupt(q->ehci, USBSTS_ERRINT);

            break;

        default:

            /* should not be triggerable */

            fprintf(stderr, "USB invalid response %d\n", p->usb_status);

            assert(0);

            break;

        }

    } else if ((p->usb_status > p->tbytes) && (p->pid == USB_TOKEN_IN)) {

        p->usb_status = USB_RET_BABBLE;

        q->qh.token |= (QTD_TOKEN_HALT | QTD_TOKEN_BABBLE);

        ehci_record_interrupt(q->ehci, USBSTS_ERRINT);

    } else {

        // TODO check 4.12 for splits



        if (p->tbytes && p->pid == USB_TOKEN_IN) {

            p->tbytes -= p->usb_status;

        } else {

            p->tbytes = 0;

        }



        DPRINTF("updating tbytes to %d\n", p->tbytes);

        set_field(&q->qh.token, p->tbytes, QTD_TOKEN_TBYTES);

    }

    ehci_finish_transfer(q, p->usb_status);

    qemu_sglist_destroy(&p->sgl);

    usb_packet_unmap(&p->packet);



    q->qh.token ^= QTD_TOKEN_DTOGGLE;

    q->qh.token &= ~QTD_TOKEN_ACTIVE;



    if (q->qh.token & QTD_TOKEN_IOC) {

        ehci_record_interrupt(q->ehci, USBSTS_INT);

    }

}
