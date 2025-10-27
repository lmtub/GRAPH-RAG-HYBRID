static void spapr_msi_write(void *opaque, hwaddr addr,

                            uint64_t data, unsigned size)

{

    sPAPRPHBState *phb = opaque;

    int ndev = addr >> 16;

    int vec = ((addr & 0xFFFF) >> 2) | data;

    uint32_t irq = phb->msi_table[ndev].irq + vec;



    trace_spapr_pci_msi_write(addr, data, irq);



    qemu_irq_pulse(xics_get_qirq(spapr->icp, irq));

}
