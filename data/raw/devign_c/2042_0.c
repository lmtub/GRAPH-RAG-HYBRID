static void spapr_phb_remove_pci_device(sPAPRDRConnector *drc,

                                        sPAPRPHBState *phb,

                                        PCIDevice *pdev,

                                        Error **errp)

{

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    drck->detach(drc, DEVICE(pdev), spapr_phb_remove_pci_device_cb, phb, errp);

}
