static int vtd_remap_irq_get(IntelIOMMUState *iommu, uint16_t index, VTDIrq *irq)

{

    VTD_IRTE irte = {};

    int ret = 0;



    ret = vtd_irte_get(iommu, index, &irte);

    if (ret) {

        return ret;

    }



    irq->trigger_mode = irte.trigger_mode;

    irq->vector = irte.vector;

    irq->delivery_mode = irte.delivery_mode;

    irq->dest = le32_to_cpu(irte.dest_id);

    if (!iommu->intr_eime) {

#define  VTD_IR_APIC_DEST_MASK         (0xff00ULL)

#define  VTD_IR_APIC_DEST_SHIFT        (8)

        irq->dest = (irq->dest & VTD_IR_APIC_DEST_MASK) >>

            VTD_IR_APIC_DEST_SHIFT;

    }

    irq->dest_mode = irte.dest_mode;

    irq->redir_hint = irte.redir_hint;



    VTD_DPRINTF(IR, "remapping interrupt index %d: trig:%u,vec:%u,"

                "deliver:%u,dest:%u,dest_mode:%u", index,

                irq->trigger_mode, irq->vector, irq->delivery_mode,

                irq->dest, irq->dest_mode);



    return 0;

}
