static void pci_spapr_set_irq(void *opaque, int irq_num, int level)

{

    /*

     * Here we use the number returned by pci_spapr_map_irq to find a

     * corresponding qemu_irq.

     */

    sPAPRPHBState *phb = opaque;



    trace_spapr_pci_lsi_set(phb->busname, irq_num, phb->lsi_table[irq_num].irq);

    qemu_set_irq(spapr_phb_lsi_qirq(phb, irq_num), level);

}
