static int ehci_state_executing(EHCIQueue *q)

{

    EHCIPacket *p = QTAILQ_FIRST(&q->packets);

    int again = 0;



    assert(p != NULL);

    assert(p->qtdaddr == q->qtdaddr);



    ehci_execute_complete(q);

    if (p->usb_status == USB_RET_ASYNC) {

        goto out;

    }

    if (p->usb_status == USB_RET_PROCERR) {

        again = -1;

        goto out;

    }



    // 4.10.3

    if (!q->async) {

        int transactCtr = get_field(q->qh.epcap, QH_EPCAP_MULT);

        transactCtr--;

        set_field(&q->qh.epcap, transactCtr, QH_EPCAP_MULT);

        // 4.10.3, bottom of page 82, should exit this state when transaction

        // counter decrements to 0

    }



    /* 4.10.5 */

    if (p->usb_status == USB_RET_NAK) {

        ehci_set_state(q->ehci, q->async, EST_HORIZONTALQH);

    } else {

        ehci_set_state(q->ehci, q->async, EST_WRITEBACK);

    }



    again = 1;



out:

    ehci_flush_qh(q);

    return again;

}
