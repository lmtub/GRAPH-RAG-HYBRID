static int spapr_populate_pci_child_dt(PCIDevice *dev, void *fdt, int offset,

                                       sPAPRPHBState *sphb)

{

    ResourceProps rp;

    bool is_bridge = false;

    int pci_status, err;

    char *buf = NULL;

    uint32_t drc_index = spapr_phb_get_pci_drc_index(sphb, dev);

    uint32_t ccode = pci_default_read_config(dev, PCI_CLASS_PROG, 3);

    uint32_t max_msi, max_msix;



    if (pci_default_read_config(dev, PCI_HEADER_TYPE, 1) ==

        PCI_HEADER_TYPE_BRIDGE) {

        is_bridge = true;

    }



    /* in accordance with PAPR+ v2.7 13.6.3, Table 181 */

    _FDT(fdt_setprop_cell(fdt, offset, "vendor-id",

                          pci_default_read_config(dev, PCI_VENDOR_ID, 2)));

    _FDT(fdt_setprop_cell(fdt, offset, "device-id",

                          pci_default_read_config(dev, PCI_DEVICE_ID, 2)));

    _FDT(fdt_setprop_cell(fdt, offset, "revision-id",

                          pci_default_read_config(dev, PCI_REVISION_ID, 1)));

    _FDT(fdt_setprop_cell(fdt, offset, "class-code", ccode));

    if (pci_default_read_config(dev, PCI_INTERRUPT_PIN, 1)) {

        _FDT(fdt_setprop_cell(fdt, offset, "interrupts",

                 pci_default_read_config(dev, PCI_INTERRUPT_PIN, 1)));

    }



    if (!is_bridge) {

        _FDT(fdt_setprop_cell(fdt, offset, "min-grant",

            pci_default_read_config(dev, PCI_MIN_GNT, 1)));

        _FDT(fdt_setprop_cell(fdt, offset, "max-latency",

            pci_default_read_config(dev, PCI_MAX_LAT, 1)));

    }



    if (pci_default_read_config(dev, PCI_SUBSYSTEM_ID, 2)) {

        _FDT(fdt_setprop_cell(fdt, offset, "subsystem-id",

                 pci_default_read_config(dev, PCI_SUBSYSTEM_ID, 2)));

    }



    if (pci_default_read_config(dev, PCI_SUBSYSTEM_VENDOR_ID, 2)) {

        _FDT(fdt_setprop_cell(fdt, offset, "subsystem-vendor-id",

                 pci_default_read_config(dev, PCI_SUBSYSTEM_VENDOR_ID, 2)));

    }



    _FDT(fdt_setprop_cell(fdt, offset, "cache-line-size",

        pci_default_read_config(dev, PCI_CACHE_LINE_SIZE, 1)));



    /* the following fdt cells are masked off the pci status register */

    pci_status = pci_default_read_config(dev, PCI_STATUS, 2);

    _FDT(fdt_setprop_cell(fdt, offset, "devsel-speed",

                          PCI_STATUS_DEVSEL_MASK & pci_status));



    if (pci_status & PCI_STATUS_FAST_BACK) {

        _FDT(fdt_setprop(fdt, offset, "fast-back-to-back", NULL, 0));

    }

    if (pci_status & PCI_STATUS_66MHZ) {

        _FDT(fdt_setprop(fdt, offset, "66mhz-capable", NULL, 0));

    }

    if (pci_status & PCI_STATUS_UDF) {

        _FDT(fdt_setprop(fdt, offset, "udf-supported", NULL, 0));

    }



    _FDT(fdt_setprop_string(fdt, offset, "name",

                            pci_find_device_name((ccode >> 16) & 0xff,

                                                 (ccode >> 8) & 0xff,

                                                 ccode & 0xff)));

    buf = spapr_phb_get_loc_code(sphb, dev);

    if (!buf) {

        error_report("Failed setting the ibm,loc-code");

        return -1;

    }



    err = fdt_setprop_string(fdt, offset, "ibm,loc-code", buf);

    g_free(buf);

    if (err < 0) {

        return err;

    }



    if (drc_index) {

        _FDT(fdt_setprop_cell(fdt, offset, "ibm,my-drc-index", drc_index));

    }



    _FDT(fdt_setprop_cell(fdt, offset, "#address-cells",

                          RESOURCE_CELLS_ADDRESS));

    _FDT(fdt_setprop_cell(fdt, offset, "#size-cells",

                          RESOURCE_CELLS_SIZE));



    max_msi = msi_nr_vectors_allocated(dev);

    if (max_msi) {

        _FDT(fdt_setprop_cell(fdt, offset, "ibm,req#msi", max_msi));

    }

    max_msix = dev->msix_entries_nr;

    if (max_msix) {

        _FDT(fdt_setprop_cell(fdt, offset, "ibm,req#msi-x", max_msix));

    }



    populate_resource_props(dev, &rp);

    _FDT(fdt_setprop(fdt, offset, "reg", (uint8_t *)rp.reg, rp.reg_len));

    _FDT(fdt_setprop(fdt, offset, "assigned-addresses",

                     (uint8_t *)rp.assigned, rp.assigned_len));



    if (sphb->pcie_ecs && pci_is_express(dev)) {

        _FDT(fdt_setprop_cell(fdt, offset, "ibm,pci-config-space-type", 0x1));

    }



    return 0;

}
