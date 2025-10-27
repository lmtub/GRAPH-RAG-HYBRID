static int xen_pt_msgdata_reg_write(XenPCIPassthroughState *s,

                                    XenPTReg *cfg_entry, uint16_t *val,

                                    uint16_t dev_value, uint16_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    XenPTMSI *msi = s->msi;

    uint16_t writable_mask = 0;

    uint16_t old_data = cfg_entry->data;

    uint32_t offset = reg->offset;



    /* check the offset whether matches the type or not */

    if (!xen_pt_msi_check_type(offset, msi->flags, DATA)) {

        /* exit I/O emulator */

        XEN_PT_ERR(&s->dev, "the offset does not match the 32/64 bit type!\n");

        return -1;

    }



    /* modify emulate register */

    writable_mask = reg->emu_mask & ~reg->ro_mask & valid_mask;

    cfg_entry->data = XEN_PT_MERGE_VALUE(*val, cfg_entry->data, writable_mask);

    /* update the msi_info too */

    msi->data = cfg_entry->data;



    /* create value for writing to I/O device register */

    *val = XEN_PT_MERGE_VALUE(*val, dev_value, 0);



    /* update MSI */

    if (cfg_entry->data != old_data) {

        if (msi->mapped) {

            xen_pt_msi_update(s);

        }

    }



    return 0;

}
