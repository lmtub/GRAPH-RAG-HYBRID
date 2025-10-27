static void xio3130_downstream_realize(PCIDevice *d, Error **errp)

{

    PCIEPort *p = PCIE_PORT(d);

    PCIESlot *s = PCIE_SLOT(d);

    int rc;



    pci_bridge_initfn(d, TYPE_PCIE_BUS);

    pcie_port_init_reg(d);



    rc = msi_init(d, XIO3130_MSI_OFFSET, XIO3130_MSI_NR_VECTOR,

                  XIO3130_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_64BIT,

                  XIO3130_MSI_SUPPORTED_FLAGS & PCI_MSI_FLAGS_MASKBIT,

                  errp);

    if (rc < 0) {

        assert(rc == -ENOTSUP);

        goto err_bridge;

    }



    rc = pci_bridge_ssvid_init(d, XIO3130_SSVID_OFFSET,

                               XIO3130_SSVID_SVID, XIO3130_SSVID_SSID,

                               errp);

    if (rc < 0) {

        goto err_bridge;

    }



    rc = pcie_cap_init(d, XIO3130_EXP_OFFSET, PCI_EXP_TYPE_DOWNSTREAM,

                       p->port, errp);

    if (rc < 0) {

        goto err_msi;

    }

    pcie_cap_flr_init(d);

    pcie_cap_deverr_init(d);

    pcie_cap_slot_init(d, s->slot);

    pcie_cap_arifwd_init(d);



    pcie_chassis_create(s->chassis);

    rc = pcie_chassis_add_slot(s);

    if (rc < 0) {


        goto err_pcie_cap;

    }



    rc = pcie_aer_init(d, PCI_ERR_VER, XIO3130_AER_OFFSET,

                       PCI_ERR_SIZEOF, errp);

    if (rc < 0) {

        goto err;

    }



    return;



err:

    pcie_chassis_del_slot(s);

err_pcie_cap:

    pcie_cap_exit(d);

err_msi:

    msi_uninit(d);

err_bridge:

    pci_bridge_exitfn(d);

}