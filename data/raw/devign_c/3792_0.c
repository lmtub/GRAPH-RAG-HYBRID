static void spapr_msi_setmsg(PCIDevice *pdev, hwaddr addr,

                             bool msix, unsigned req_num)

{

    unsigned i;

    MSIMessage msg = { .address = addr, .data = 0 };



    if (!msix) {

        msi_set_message(pdev, msg);

        trace_spapr_pci_msi_setup(pdev->name, 0, msg.address);

        return;

    }



    for (i = 0; i < req_num; ++i) {

        msg.address = addr | (i << 2);

        msix_set_message(pdev, i, msg);

        trace_spapr_pci_msi_setup(pdev->name, i, msg.address);

    }

}
