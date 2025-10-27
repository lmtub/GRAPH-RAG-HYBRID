static void mptsas_update_interrupt(MPTSASState *s)

{

    PCIDevice *pci = (PCIDevice *) s;

    uint32_t state = s->intr_status & ~(s->intr_mask | MPI_HIS_IOP_DOORBELL_STATUS);



    if (s->msi_in_use && msi_enabled(pci)) {

        if (state) {

            trace_mptsas_irq_msi(s);

            msi_notify(pci, 0);

        }

    }



    trace_mptsas_irq_intx(s, !!state);

    pci_set_irq(pci, !!state);

}
