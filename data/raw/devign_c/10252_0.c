static int xen_pt_msixctrl_reg_init(XenPCIPassthroughState *s,

                                    XenPTRegInfo *reg, uint32_t real_offset,

                                    uint32_t *data)

{

    PCIDevice *d = &s->dev;

    uint16_t reg_field = 0;



    /* use I/O device register's value as initial value */

    reg_field = pci_get_word(d->config + real_offset);



    if (reg_field & PCI_MSIX_FLAGS_ENABLE) {

        XEN_PT_LOG(d, "MSIX already enabled, disabling it first\n");

        xen_host_pci_set_word(&s->real_device, real_offset,

                              reg_field & ~PCI_MSIX_FLAGS_ENABLE);

    }



    s->msix->ctrl_offset = real_offset;



    *data = reg->init_val;

    return 0;

}
