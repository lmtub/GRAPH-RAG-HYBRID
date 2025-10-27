static int xen_pt_bar_reg_write(XenPCIPassthroughState *s, XenPTReg *cfg_entry,

                                uint32_t *val, uint32_t dev_value,

                                uint32_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    XenPTRegion *base = NULL;

    PCIDevice *d = &s->dev;

    const PCIIORegion *r;

    uint32_t writable_mask = 0;

    uint32_t bar_emu_mask = 0;

    uint32_t bar_ro_mask = 0;

    uint32_t r_size = 0;

    int index = 0;



    index = xen_pt_bar_offset_to_index(reg->offset);

    if (index < 0 || index >= PCI_NUM_REGIONS) {

        XEN_PT_ERR(d, "Internal error: Invalid BAR index [%d].\n", index);

        return -1;

    }



    r = &d->io_regions[index];

    base = &s->bases[index];

    r_size = xen_pt_get_emul_size(base->bar_flag, r->size);



    /* set emulate mask and read-only mask values depend on the BAR flag */

    switch (s->bases[index].bar_flag) {

    case XEN_PT_BAR_FLAG_MEM:

        bar_emu_mask = XEN_PT_BAR_MEM_EMU_MASK;

        if (!r_size) {

            /* low 32 bits mask for 64 bit bars */

            bar_ro_mask = XEN_PT_BAR_ALLF;

        } else {

            bar_ro_mask = XEN_PT_BAR_MEM_RO_MASK | (r_size - 1);

        }

        break;

    case XEN_PT_BAR_FLAG_IO:

        bar_emu_mask = XEN_PT_BAR_IO_EMU_MASK;

        bar_ro_mask = XEN_PT_BAR_IO_RO_MASK | (r_size - 1);

        break;

    case XEN_PT_BAR_FLAG_UPPER:

        bar_emu_mask = XEN_PT_BAR_ALLF;

        bar_ro_mask = r_size ? r_size - 1 : 0;

        break;

    default:

        break;

    }



    /* modify emulate register */

    writable_mask = bar_emu_mask & ~bar_ro_mask & valid_mask;

    cfg_entry->data = XEN_PT_MERGE_VALUE(*val, cfg_entry->data, writable_mask);



    /* check whether we need to update the virtual region address or not */

    switch (s->bases[index].bar_flag) {

    case XEN_PT_BAR_FLAG_UPPER:

    case XEN_PT_BAR_FLAG_MEM:

        /* nothing to do */

        break;

    case XEN_PT_BAR_FLAG_IO:

        /* nothing to do */

        break;

    default:

        break;

    }



    /* create value for writing to I/O device register */

    *val = XEN_PT_MERGE_VALUE(*val, dev_value, 0);



    return 0;

}
