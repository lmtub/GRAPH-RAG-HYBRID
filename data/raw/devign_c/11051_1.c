static void ehci_trace_itd(EHCIState *s, target_phys_addr_t addr, EHCIitd *itd)

{

    trace_usb_ehci_itd(addr, itd->next);

}
