qemu_irq xics_assign_irq(struct icp_state *icp, int irq,

                         enum xics_irq_type type)

{

    if ((irq < icp->ics->offset)

        || (irq >= (icp->ics->offset + icp->ics->nr_irqs))) {

        return NULL;

    }



    assert((type == XICS_MSI) || (type == XICS_LSI));



    icp->ics->irqs[irq - icp->ics->offset].type = type;

    return icp->ics->qirqs[irq - icp->ics->offset];

}
