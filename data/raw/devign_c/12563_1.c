static int assigned_device_pci_cap_init(PCIDevice *pci_dev, Error **errp)

{

    AssignedDevice *dev = DO_UPCAST(AssignedDevice, dev, pci_dev);

    PCIRegion *pci_region = dev->real_device.regions;

    int ret, pos;

    Error *local_err = NULL;



    /* Clear initial capabilities pointer and status copied from hw */

    pci_set_byte(pci_dev->config + PCI_CAPABILITY_LIST, 0);

    pci_set_word(pci_dev->config + PCI_STATUS,

                 pci_get_word(pci_dev->config + PCI_STATUS) &

                 ~PCI_STATUS_CAP_LIST);



    /* Expose MSI capability

     * MSI capability is the 1st capability in capability config */

    pos = pci_find_cap_offset(pci_dev, PCI_CAP_ID_MSI, 0);

    if (pos != 0 && kvm_check_extension(kvm_state, KVM_CAP_ASSIGN_DEV_IRQ)) {

        verify_irqchip_in_kernel(&local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return -ENOTSUP;

        }

        dev->cap.available |= ASSIGNED_DEVICE_CAP_MSI;

        /* Only 32-bit/no-mask currently supported */

        ret = pci_add_capability2(pci_dev, PCI_CAP_ID_MSI, pos, 10,

                                  &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            return ret;

        }

        pci_dev->msi_cap = pos;



        pci_set_word(pci_dev->config + pos + PCI_MSI_FLAGS,

                     pci_get_word(pci_dev->config + pos + PCI_MSI_FLAGS) &

                     PCI_MSI_FLAGS_QMASK);

        pci_set_long(pci_dev->config + pos + PCI_MSI_ADDRESS_LO, 0);

        pci_set_word(pci_dev->config + pos + PCI_MSI_DATA_32, 0);



        /* Set writable fields */

        pci_set_word(pci_dev->wmask + pos + PCI_MSI_FLAGS,

                     PCI_MSI_FLAGS_QSIZE | PCI_MSI_FLAGS_ENABLE);

        pci_set_long(pci_dev->wmask + pos + PCI_MSI_ADDRESS_LO, 0xfffffffc);

        pci_set_word(pci_dev->wmask + pos + PCI_MSI_DATA_32, 0xffff);

    }

    /* Expose MSI-X capability */

    pos = pci_find_cap_offset(pci_dev, PCI_CAP_ID_MSIX, 0);

    if (pos != 0 && kvm_device_msix_supported(kvm_state)) {

        int bar_nr;

        uint32_t msix_table_entry;



        verify_irqchip_in_kernel(&local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return -ENOTSUP;

        }

        dev->cap.available |= ASSIGNED_DEVICE_CAP_MSIX;

        ret = pci_add_capability2(pci_dev, PCI_CAP_ID_MSIX, pos, 12,

                                  &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            return ret;

        }

        pci_dev->msix_cap = pos;



        pci_set_word(pci_dev->config + pos + PCI_MSIX_FLAGS,

                     pci_get_word(pci_dev->config + pos + PCI_MSIX_FLAGS) &

                     PCI_MSIX_FLAGS_QSIZE);



        /* Only enable and function mask bits are writable */

        pci_set_word(pci_dev->wmask + pos + PCI_MSIX_FLAGS,

                     PCI_MSIX_FLAGS_ENABLE | PCI_MSIX_FLAGS_MASKALL);



        msix_table_entry = pci_get_long(pci_dev->config + pos + PCI_MSIX_TABLE);

        bar_nr = msix_table_entry & PCI_MSIX_FLAGS_BIRMASK;

        msix_table_entry &= ~PCI_MSIX_FLAGS_BIRMASK;

        dev->msix_table_addr = pci_region[bar_nr].base_addr + msix_table_entry;

        dev->msix_max = pci_get_word(pci_dev->config + pos + PCI_MSIX_FLAGS);

        dev->msix_max &= PCI_MSIX_FLAGS_QSIZE;

        dev->msix_max += 1;

    }



    /* Minimal PM support, nothing writable, device appears to NAK changes */

    pos = pci_find_cap_offset(pci_dev, PCI_CAP_ID_PM, 0);

    if (pos) {

        uint16_t pmc;



        ret = pci_add_capability2(pci_dev, PCI_CAP_ID_PM, pos, PCI_PM_SIZEOF,

                                  &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            return ret;

        }



        assigned_dev_setup_cap_read(dev, pos, PCI_PM_SIZEOF);



        pmc = pci_get_word(pci_dev->config + pos + PCI_CAP_FLAGS);

        pmc &= (PCI_PM_CAP_VER_MASK | PCI_PM_CAP_DSI);

        pci_set_word(pci_dev->config + pos + PCI_CAP_FLAGS, pmc);



        /* assign_device will bring the device up to D0, so we don't need

         * to worry about doing that ourselves here. */

        pci_set_word(pci_dev->config + pos + PCI_PM_CTRL,

                     PCI_PM_CTRL_NO_SOFT_RESET);



        pci_set_byte(pci_dev->config + pos + PCI_PM_PPB_EXTENSIONS, 0);

        pci_set_byte(pci_dev->config + pos + PCI_PM_DATA_REGISTER, 0);

    }



    pos = pci_find_cap_offset(pci_dev, PCI_CAP_ID_EXP, 0);

    if (pos) {

        uint8_t version, size = 0;

        uint16_t type, devctl, lnksta;

        uint32_t devcap, lnkcap;



        version = pci_get_byte(pci_dev->config + pos + PCI_EXP_FLAGS);

        version &= PCI_EXP_FLAGS_VERS;

        if (version == 1) {

            size = 0x14;

        } else if (version == 2) {

            /*

             * Check for non-std size, accept reduced size to 0x34,

             * which is what bcm5761 implemented, violating the

             * PCIe v3.0 spec that regs should exist and be read as 0,

             * not optionally provided and shorten the struct size.

             */

            size = MIN(0x3c, PCI_CONFIG_SPACE_SIZE - pos);

            if (size < 0x34) {

                error_setg(errp, "Invalid size PCIe cap-id 0x%x",

                           PCI_CAP_ID_EXP);

                return -EINVAL;

            } else if (size != 0x3c) {

                error_report("WARNING, %s: PCIe cap-id 0x%x has "

                             "non-standard size 0x%x; std size should be 0x3c",

                             __func__, PCI_CAP_ID_EXP, size);

            }

        } else if (version == 0) {

            uint16_t vid, did;

            vid = pci_get_word(pci_dev->config + PCI_VENDOR_ID);

            did = pci_get_word(pci_dev->config + PCI_DEVICE_ID);

            if (vid == PCI_VENDOR_ID_INTEL && did == 0x10ed) {

                /*

                 * quirk for Intel 82599 VF with invalid PCIe capability

                 * version, should really be version 2 (same as PF)

                 */

                size = 0x3c;

            }

        }



        if (size == 0) {

            error_setg(errp, "Unsupported PCI express capability version %d",

                       version);

            return -EINVAL;

        }



        ret = pci_add_capability2(pci_dev, PCI_CAP_ID_EXP, pos, size,

                                  &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            return ret;

        }



        assigned_dev_setup_cap_read(dev, pos, size);



        type = pci_get_word(pci_dev->config + pos + PCI_EXP_FLAGS);

        type = (type & PCI_EXP_FLAGS_TYPE) >> 4;

        if (type != PCI_EXP_TYPE_ENDPOINT &&

            type != PCI_EXP_TYPE_LEG_END && type != PCI_EXP_TYPE_RC_END) {

            error_setg(errp, "Device assignment only supports endpoint "

                       "assignment, device type %d", type);

            return -EINVAL;

        }



        /* capabilities, pass existing read-only copy

         * PCI_EXP_FLAGS_IRQ: updated by hardware, should be direct read */



        /* device capabilities: hide FLR */

        devcap = pci_get_long(pci_dev->config + pos + PCI_EXP_DEVCAP);

        devcap &= ~PCI_EXP_DEVCAP_FLR;

        pci_set_long(pci_dev->config + pos + PCI_EXP_DEVCAP, devcap);



        /* device control: clear all error reporting enable bits, leaving

         *                 only a few host values.  Note, these are

         *                 all writable, but not passed to hw.

         */

        devctl = pci_get_word(pci_dev->config + pos + PCI_EXP_DEVCTL);

        devctl = (devctl & (PCI_EXP_DEVCTL_READRQ | PCI_EXP_DEVCTL_PAYLOAD)) |

                  PCI_EXP_DEVCTL_RELAX_EN | PCI_EXP_DEVCTL_NOSNOOP_EN;

        pci_set_word(pci_dev->config + pos + PCI_EXP_DEVCTL, devctl);

        devctl = PCI_EXP_DEVCTL_BCR_FLR | PCI_EXP_DEVCTL_AUX_PME;

        pci_set_word(pci_dev->wmask + pos + PCI_EXP_DEVCTL, ~devctl);



        /* Clear device status */

        pci_set_word(pci_dev->config + pos + PCI_EXP_DEVSTA, 0);



        /* Link capabilities, expose links and latencues, clear reporting */

        lnkcap = pci_get_long(pci_dev->config + pos + PCI_EXP_LNKCAP);

        lnkcap &= (PCI_EXP_LNKCAP_SLS | PCI_EXP_LNKCAP_MLW |

                   PCI_EXP_LNKCAP_ASPMS | PCI_EXP_LNKCAP_L0SEL |

                   PCI_EXP_LNKCAP_L1EL);

        pci_set_long(pci_dev->config + pos + PCI_EXP_LNKCAP, lnkcap);



        /* Link control, pass existing read-only copy.  Should be writable? */



        /* Link status, only expose current speed and width */

        lnksta = pci_get_word(pci_dev->config + pos + PCI_EXP_LNKSTA);

        lnksta &= (PCI_EXP_LNKSTA_CLS | PCI_EXP_LNKSTA_NLW);

        pci_set_word(pci_dev->config + pos + PCI_EXP_LNKSTA, lnksta);



        if (version >= 2) {

            /* Slot capabilities, control, status - not needed for endpoints */

            pci_set_long(pci_dev->config + pos + PCI_EXP_SLTCAP, 0);

            pci_set_word(pci_dev->config + pos + PCI_EXP_SLTCTL, 0);

            pci_set_word(pci_dev->config + pos + PCI_EXP_SLTSTA, 0);



            /* Root control, capabilities, status - not needed for endpoints */

            pci_set_word(pci_dev->config + pos + PCI_EXP_RTCTL, 0);

            pci_set_word(pci_dev->config + pos + PCI_EXP_RTCAP, 0);

            pci_set_long(pci_dev->config + pos + PCI_EXP_RTSTA, 0);



            /* Device capabilities/control 2, pass existing read-only copy */

            /* Link control 2, pass existing read-only copy */

        }

    }



    pos = pci_find_cap_offset(pci_dev, PCI_CAP_ID_PCIX, 0);

    if (pos) {

        uint16_t cmd;

        uint32_t status;



        /* Only expose the minimum, 8 byte capability */

        ret = pci_add_capability2(pci_dev, PCI_CAP_ID_PCIX, pos, 8,

                                  &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            return ret;

        }



        assigned_dev_setup_cap_read(dev, pos, 8);



        /* Command register, clear upper bits, including extended modes */

        cmd = pci_get_word(pci_dev->config + pos + PCI_X_CMD);

        cmd &= (PCI_X_CMD_DPERR_E | PCI_X_CMD_ERO | PCI_X_CMD_MAX_READ |

                PCI_X_CMD_MAX_SPLIT);

        pci_set_word(pci_dev->config + pos + PCI_X_CMD, cmd);



        /* Status register, update with emulated PCI bus location, clear

         * error bits, leave the rest. */

        status = pci_get_long(pci_dev->config + pos + PCI_X_STATUS);

        status &= ~(PCI_X_STATUS_BUS | PCI_X_STATUS_DEVFN);

        status |= (pci_bus_num(pci_dev->bus) << 8) | pci_dev->devfn;

        status &= ~(PCI_X_STATUS_SPL_DISC | PCI_X_STATUS_UNX_SPL |

                    PCI_X_STATUS_SPL_ERR);

        pci_set_long(pci_dev->config + pos + PCI_X_STATUS, status);

    }



    pos = pci_find_cap_offset(pci_dev, PCI_CAP_ID_VPD, 0);

    if (pos) {

        /* Direct R/W passthrough */

        ret = pci_add_capability2(pci_dev, PCI_CAP_ID_VPD, pos, 8,

                                  &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            return ret;

        }



        assigned_dev_setup_cap_read(dev, pos, 8);



        /* direct write for cap content */

        assigned_dev_direct_config_write(dev, pos + 2, 6);

    }



    /* Devices can have multiple vendor capabilities, get them all */

    for (pos = 0; (pos = pci_find_cap_offset(pci_dev, PCI_CAP_ID_VNDR, pos));

        pos += PCI_CAP_LIST_NEXT) {

        uint8_t len = pci_get_byte(pci_dev->config + pos + PCI_CAP_FLAGS);

        /* Direct R/W passthrough */

        ret = pci_add_capability2(pci_dev, PCI_CAP_ID_VNDR, pos, len,

                                  &local_err);

        if (ret < 0) {

            error_propagate(errp, local_err);

            return ret;

        }



        assigned_dev_setup_cap_read(dev, pos, len);



        /* direct write for cap content */

        assigned_dev_direct_config_write(dev, pos + 2, len - 2);

    }



    /* If real and virtual capability list status bits differ, virtualize the

     * access. */

    if ((pci_get_word(pci_dev->config + PCI_STATUS) & PCI_STATUS_CAP_LIST) !=

        (assigned_dev_pci_read_byte(pci_dev, PCI_STATUS) &

         PCI_STATUS_CAP_LIST)) {

        dev->emulate_config_read[PCI_STATUS] |= PCI_STATUS_CAP_LIST;

    }



    return 0;

}
