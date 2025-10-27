static inline int xhci_running(XHCIState *xhci)

{

    return !(xhci->usbsts & USBSTS_HCH) && !xhci->intr[0].er_full;

}
