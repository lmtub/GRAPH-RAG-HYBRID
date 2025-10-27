static void xhci_intr_raise(XHCIState *xhci, int v)

{

    PCIDevice *pci_dev = PCI_DEVICE(xhci);




    xhci->intr[v].erdp_low |= ERDP_EHB;

    xhci->intr[v].iman |= IMAN_IP;

    xhci->usbsts |= USBSTS_EINT;






    if (!(xhci->intr[v].iman & IMAN_IE)) {





    if (!(xhci->usbcmd & USBCMD_INTE)) {





    if (msix_enabled(pci_dev)) {

        trace_usb_xhci_irq_msix(v);

        msix_notify(pci_dev, v);





    if (msi_enabled(pci_dev)) {

        trace_usb_xhci_irq_msi(v);

        msi_notify(pci_dev, v);





    if (v == 0) {

        trace_usb_xhci_irq_intx(1);

        pci_irq_assert(pci_dev);

