static void xen_pt_pci_write_config(PCIDevice *d, uint32_t addr,

                                    uint32_t val, int len)

{

    XenPCIPassthroughState *s = DO_UPCAST(XenPCIPassthroughState, dev, d);

    int index = 0;

    XenPTRegGroup *reg_grp_entry = NULL;

    int rc = 0;

    uint32_t read_val = 0, wb_mask;

    int emul_len = 0;

    XenPTReg *reg_entry = NULL;

    uint32_t find_addr = addr;

    XenPTRegInfo *reg = NULL;



    if (xen_pt_pci_config_access_check(d, addr, len)) {

        return;

    }



    XEN_PT_LOG_CONFIG(d, addr, val, len);



    /* check unused BAR register */

    index = xen_pt_bar_offset_to_index(addr);

    if ((index >= 0) && (val > 0 && val < XEN_PT_BAR_ALLF) &&

        (s->bases[index].bar_flag == XEN_PT_BAR_FLAG_UNUSED)) {

        XEN_PT_WARN(d, "Guest attempt to set address to unused Base Address "

                    "Register. (addr: 0x%02x, len: %d)\n", addr, len);

    }



    /* find register group entry */

    reg_grp_entry = xen_pt_find_reg_grp(s, addr);

    if (reg_grp_entry) {

        /* check 0-Hardwired register group */

        if (reg_grp_entry->reg_grp->grp_type == XEN_PT_GRP_TYPE_HARDWIRED) {

            /* ignore silently */

            XEN_PT_WARN(d, "Access to 0-Hardwired register. "

                        "(addr: 0x%02x, len: %d)\n", addr, len);

            return;

        }

    }



    rc = xen_host_pci_get_block(&s->real_device, addr,

                                (uint8_t *)&read_val, len);

    if (rc < 0) {

        XEN_PT_ERR(d, "pci_read_block failed. return value: %d.\n", rc);

        memset(&read_val, 0xff, len);

        wb_mask = 0;

    } else {

        wb_mask = 0xFFFFFFFF >> ((4 - len) << 3);

    }



    /* pass directly to the real device for passthrough type register group */

    if (reg_grp_entry == NULL) {

        goto out;

    }



    memory_region_transaction_begin();

    pci_default_write_config(d, addr, val, len);



    /* adjust the read and write value to appropriate CFC-CFF window */

    read_val <<= (addr & 3) << 3;

    val <<= (addr & 3) << 3;

    emul_len = len;



    /* loop around the guest requested size */

    while (emul_len > 0) {

        /* find register entry to be emulated */

        reg_entry = xen_pt_find_reg(reg_grp_entry, find_addr);

        if (reg_entry) {

            reg = reg_entry->reg;

            uint32_t real_offset = reg_grp_entry->base_offset + reg->offset;

            uint32_t valid_mask = 0xFFFFFFFF >> ((4 - emul_len) << 3);

            uint8_t *ptr_val = NULL;



            valid_mask <<= (find_addr - real_offset) << 3;

            ptr_val = (uint8_t *)&val + (real_offset & 3);

            if (reg->emu_mask == (0xFFFFFFFF >> ((4 - reg->size) << 3))) {

                wb_mask &= ~((reg->emu_mask

                              >> ((find_addr - real_offset) << 3))

                             << ((len - emul_len) << 3));

            }



            /* do emulation based on register size */

            switch (reg->size) {

            case 1:

                if (reg->u.b.write) {

                    rc = reg->u.b.write(s, reg_entry, ptr_val,

                                        read_val >> ((real_offset & 3) << 3),

                                        valid_mask);

                }

                break;

            case 2:

                if (reg->u.w.write) {

                    rc = reg->u.w.write(s, reg_entry, (uint16_t *)ptr_val,

                                        (read_val >> ((real_offset & 3) << 3)),

                                        valid_mask);

                }

                break;

            case 4:

                if (reg->u.dw.write) {

                    rc = reg->u.dw.write(s, reg_entry, (uint32_t *)ptr_val,

                                         (read_val >> ((real_offset & 3) << 3)),

                                         valid_mask);

                }

                break;

            }



            if (rc < 0) {

                xen_shutdown_fatal_error("Internal error: Invalid write"

                                         " emulation. (%s, rc: %d)\n",

                                         __func__, rc);

                return;

            }



            /* calculate next address to find */

            emul_len -= reg->size;

            if (emul_len > 0) {

                find_addr = real_offset + reg->size;

            }

        } else {

            /* nothing to do with passthrough type register,

             * continue to find next byte */

            emul_len--;

            find_addr++;

        }

    }



    /* need to shift back before passing them to xen_host_pci_device */

    val >>= (addr & 3) << 3;



    memory_region_transaction_commit();



out:

    for (index = 0; wb_mask; index += len) {

        /* unknown regs are passed through */

        while (!(wb_mask & 0xff)) {

            index++;

            wb_mask >>= 8;

        }

        len = 0;

        do {

            len++;

            wb_mask >>= 8;

        } while (wb_mask & 0xff);

        rc = xen_host_pci_set_block(&s->real_device, addr + index,

                                    (uint8_t *)&val + index, len);



        if (rc < 0) {

            XEN_PT_ERR(d, "pci_write_block failed. return value: %d.\n", rc);

        }

    }

}
