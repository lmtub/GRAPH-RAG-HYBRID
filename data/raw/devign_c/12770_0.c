static void pxb_dev_exitfn(PCIDevice *pci_dev)

{

    PXBDev *pxb = PXB_DEV(pci_dev);



    pxb_dev_list = g_list_remove(pxb_dev_list, pxb);

}
