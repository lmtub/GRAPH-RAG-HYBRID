static void ehci_trace_qh(EHCIQueue *q, target_phys_addr_t addr, EHCIqh *qh)

{

    trace_usb_ehci_qh(q, addr, qh->next,

                      qh->current_qtd, qh->next_qtd, qh->altnext_qtd,

                      get_field(qh->epchar, QH_EPCHAR_RL),

                      get_field(qh->epchar, QH_EPCHAR_MPLEN),

                      get_field(qh->epchar, QH_EPCHAR_EPS),

                      get_field(qh->epchar, QH_EPCHAR_EP),

                      get_field(qh->epchar, QH_EPCHAR_DEVADDR),

                      (bool)(qh->epchar & QH_EPCHAR_C),

                      (bool)(qh->epchar & QH_EPCHAR_H),

                      (bool)(qh->epchar & QH_EPCHAR_DTC),

                      (bool)(qh->epchar & QH_EPCHAR_I));

}
