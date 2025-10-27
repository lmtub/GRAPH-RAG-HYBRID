static int vtd_interrupt_remap_msi(IntelIOMMUState *iommu,

                                   MSIMessage *origin,

                                   MSIMessage *translated)

{

    int ret = 0;

    VTD_IR_MSIAddress addr;

    uint16_t index;

    VTDIrq irq = {0};



    assert(origin && translated);



    if (!iommu || !iommu->intr_enabled) {

        goto do_not_translate;

    }



    if (origin->address & VTD_MSI_ADDR_HI_MASK) {

        VTD_DPRINTF(GENERAL, "error: MSI addr high 32 bits nonzero"

                    " during interrupt remapping: 0x%"PRIx32,

                    (uint32_t)((origin->address & VTD_MSI_ADDR_HI_MASK) >> \

                    VTD_MSI_ADDR_HI_SHIFT));

        return -VTD_FR_IR_REQ_RSVD;

    }



    addr.data = origin->address & VTD_MSI_ADDR_LO_MASK;

    if (le16_to_cpu(addr.__head) != 0xfee) {

        VTD_DPRINTF(GENERAL, "error: MSI addr low 32 bits invalid: "

                    "0x%"PRIx32, addr.data);

        return -VTD_FR_IR_REQ_RSVD;

    }



    /* This is compatible mode. */

    if (addr.int_mode != VTD_IR_INT_FORMAT_REMAP) {

        goto do_not_translate;

    }



    index = addr.index_h << 15 | le16_to_cpu(addr.index_l);



#define  VTD_IR_MSI_DATA_SUBHANDLE       (0x0000ffff)

#define  VTD_IR_MSI_DATA_RESERVED        (0xffff0000)



    if (addr.sub_valid) {

        /* See VT-d spec 5.1.2.2 and 5.1.3 on subhandle */

        index += origin->data & VTD_IR_MSI_DATA_SUBHANDLE;

    }



    ret = vtd_remap_irq_get(iommu, index, &irq);

    if (ret) {

        return ret;

    }



    if (addr.sub_valid) {

        VTD_DPRINTF(IR, "received MSI interrupt");

        if (origin->data & VTD_IR_MSI_DATA_RESERVED) {

            VTD_DPRINTF(GENERAL, "error: MSI data bits non-zero for "

                        "interrupt remappable entry: 0x%"PRIx32,

                        origin->data);

            return -VTD_FR_IR_REQ_RSVD;

        }

    } else {

        uint8_t vector = origin->data & 0xff;

        VTD_DPRINTF(IR, "received IOAPIC interrupt");

        /* IOAPIC entry vector should be aligned with IRTE vector

         * (see vt-d spec 5.1.5.1). */

        if (vector != irq.vector) {

            VTD_DPRINTF(GENERAL, "IOAPIC vector inconsistent: "

                        "entry: %d, IRTE: %d, index: %d",

                        vector, irq.vector, index);

        }

    }



    /*

     * We'd better keep the last two bits, assuming that guest OS

     * might modify it. Keep it does not hurt after all.

     */

    irq.msi_addr_last_bits = addr.__not_care;



    /* Translate VTDIrq to MSI message */

    vtd_generate_msi_message(&irq, translated);



    VTD_DPRINTF(IR, "mapping MSI 0x%"PRIx64":0x%"PRIx32 " -> "

                "0x%"PRIx64":0x%"PRIx32, origin->address, origin->data,

                translated->address, translated->data);

    return 0;



do_not_translate:

    memcpy(translated, origin, sizeof(*origin));

    return 0;

}
