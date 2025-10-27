static int xen_pt_msixctrl_reg_write(XenPCIPassthroughState *s,

                                     XenPTReg *cfg_entry, uint16_t *val,

                                     uint16_t dev_value, uint16_t valid_mask)

{

    XenPTRegInfo *reg = cfg_entry->reg;

    uint16_t writable_mask = 0;

    uint16_t throughable_mask = get_throughable_mask(s, reg, valid_mask);

    int debug_msix_enabled_old;



    /* modify emulate register */

    writable_mask = reg->emu_mask & ~reg->ro_mask & valid_mask;

    cfg_entry->data = XEN_PT_MERGE_VALUE(*val, cfg_entry->data, writable_mask);



    /* create value for writing to I/O device register */

    *val = XEN_PT_MERGE_VALUE(*val, dev_value, throughable_mask);



    /* update MSI-X */

    if ((*val & PCI_MSIX_FLAGS_ENABLE)

        && !(*val & PCI_MSIX_FLAGS_MASKALL)) {

        xen_pt_msix_update(s);

    } else if (!(*val & PCI_MSIX_FLAGS_ENABLE) && s->msix->enabled) {

        xen_pt_msix_disable(s);

    }



    debug_msix_enabled_old = s->msix->enabled;

    s->msix->enabled = !!(*val & PCI_MSIX_FLAGS_ENABLE);

    if (s->msix->enabled != debug_msix_enabled_old) {

        XEN_PT_LOG(&s->dev, "%s MSI-X\n",

                   s->msix->enabled ? "enable" : "disable");

    }



    return 0;

}
