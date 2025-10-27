static sPAPRDRConnector *spapr_phb_get_pci_drc(sPAPRPHBState *phb,

                                               PCIDevice *pdev)

{

    uint32_t busnr = pci_bus_num(PCI_BUS(qdev_get_parent_bus(DEVICE(pdev))));

    return spapr_dr_connector_by_id(SPAPR_DR_CONNECTOR_TYPE_PCI,

                                    (phb->index << 16) |

                                    (busnr << 8) |

                                    pdev->devfn);

}
