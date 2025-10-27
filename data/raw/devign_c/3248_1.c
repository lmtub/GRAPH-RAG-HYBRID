static int xen_pt_config_reg_init(XenPCIPassthroughState *s,

                                  XenPTRegGroup *reg_grp, XenPTRegInfo *reg)

{

    XenPTReg *reg_entry;

    uint32_t data = 0;

    int rc = 0;



    reg_entry = g_new0(XenPTReg, 1);

    reg_entry->reg = reg;



    if (reg->init) {

        /* initialize emulate register */

        rc = reg->init(s, reg_entry->reg,

                       reg_grp->base_offset + reg->offset, &data);

        if (rc < 0) {

            free(reg_entry);

            return rc;

        }

        if (data == XEN_PT_INVALID_REG) {

            /* free unused BAR register entry */

            free(reg_entry);

            return 0;

        }

        /* set register value */

        reg_entry->data = data;

    }

    /* list add register entry */

    QLIST_INSERT_HEAD(&reg_grp->reg_tbl_list, reg_entry, entries);



    return 0;

}
