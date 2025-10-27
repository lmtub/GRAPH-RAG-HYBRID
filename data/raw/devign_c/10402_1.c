static void xhci_runtime_write(void *ptr, hwaddr reg,

                               uint64_t val, unsigned size)

{

    XHCIState *xhci = ptr;

    int v = (reg - 0x20) / 0x20;

    XHCIInterrupter *intr = &xhci->intr[v];

    trace_usb_xhci_runtime_write(reg, val);



    if (reg < 0x20) {

        trace_usb_xhci_unimplemented("runtime write", reg);

        return;




    switch (reg & 0x1f) {

    case 0x00: /* IMAN */

        if (val & IMAN_IP) {

            intr->iman &= ~IMAN_IP;


        intr->iman &= ~IMAN_IE;

        intr->iman |= val & IMAN_IE;

        if (v == 0) {

            xhci_intx_update(xhci);


        xhci_msix_update(xhci, v);

        break;

    case 0x04: /* IMOD */

        intr->imod = val;

        break;

    case 0x08: /* ERSTSZ */

        intr->erstsz = val & 0xffff;

        break;

    case 0x10: /* ERSTBA low */

        /* XXX NEC driver bug: it doesn't align this to 64 bytes

        intr->erstba_low = val & 0xffffffc0; */

        intr->erstba_low = val & 0xfffffff0;

        break;

    case 0x14: /* ERSTBA high */

        intr->erstba_high = val;

        xhci_er_reset(xhci, v);

        break;

    case 0x18: /* ERDP low */


            intr->erdp_low &= ~ERDP_EHB;


        intr->erdp_low = (val & ~ERDP_EHB) | (intr->erdp_low & ERDP_EHB);










        break;

    case 0x1c: /* ERDP high */

        intr->erdp_high = val;

        xhci_events_update(xhci, v);

        break;

    default:

        trace_usb_xhci_unimplemented("oper write", reg);

