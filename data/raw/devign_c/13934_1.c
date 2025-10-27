static void ohci_reset(OHCIState *ohci)

{

    OHCIPort *port;

    int i;



    ohci->ctl = 0;

    ohci->old_ctl = 0;

    ohci->status = 0;

    ohci->intr_status = 0;

    ohci->intr = OHCI_INTR_MIE;



    ohci->hcca = 0;

    ohci->ctrl_head = ohci->ctrl_cur = 0;

    ohci->bulk_head = ohci->bulk_cur = 0;

    ohci->per_cur = 0;

    ohci->done = 0;

    ohci->done_count = 7;



    /* FSMPS is marked TBD in OCHI 1.0, what gives ffs?

     * I took the value linux sets ...

     */

    ohci->fsmps = 0x2778;

    ohci->fi = 0x2edf;

    ohci->fit = 0;

    ohci->frt = 0;

    ohci->frame_number = 0;

    ohci->pstart = 0;

    ohci->lst = OHCI_LS_THRESH;



    ohci->rhdesc_a = OHCI_RHA_NPS | ohci->num_ports;

    ohci->rhdesc_b = 0x0; /* Impl. specific */

    ohci->rhstatus = 0;



    for (i = 0; i < ohci->num_ports; i++)

      {

        port = &ohci->rhport[i];

        port->ctrl = 0;

        if (port->port.dev)

            ohci_attach(&port->port, port->port.dev);

      }

    if (ohci->async_td) {

        usb_cancel_packet(&ohci->usb_packet);

        ohci->async_td = 0;

    }

    dprintf("usb-ohci: Reset %s\n", ohci->name);

}
