int pcie_aer_init(PCIDevice *dev, uint16_t offset, uint16_t size)

{

    PCIExpressDevice *exp;



    pcie_add_capability(dev, PCI_EXT_CAP_ID_ERR, PCI_ERR_VER,

                        offset, size);

    exp = &dev->exp;

    exp->aer_cap = offset;



    /* log_max is property */

    if (dev->exp.aer_log.log_max == PCIE_AER_LOG_MAX_UNSET) {

        dev->exp.aer_log.log_max = PCIE_AER_LOG_MAX_DEFAULT;

    }

    /* clip down the value to avoid unreasobale memory usage */

    if (dev->exp.aer_log.log_max > PCIE_AER_LOG_MAX_LIMIT) {

        return -EINVAL;

    }

    dev->exp.aer_log.log = g_malloc0(sizeof dev->exp.aer_log.log[0] *

                                        dev->exp.aer_log.log_max);



    pci_set_long(dev->w1cmask + offset + PCI_ERR_UNCOR_STATUS,

                 PCI_ERR_UNC_SUPPORTED);



    pci_set_long(dev->config + offset + PCI_ERR_UNCOR_SEVER,

                 PCI_ERR_UNC_SEVERITY_DEFAULT);

    pci_set_long(dev->wmask + offset + PCI_ERR_UNCOR_SEVER,

                 PCI_ERR_UNC_SUPPORTED);



    pci_long_test_and_set_mask(dev->w1cmask + offset + PCI_ERR_COR_STATUS,

                               PCI_ERR_COR_SUPPORTED);



    pci_set_long(dev->config + offset + PCI_ERR_COR_MASK,

                 PCI_ERR_COR_MASK_DEFAULT);

    pci_set_long(dev->wmask + offset + PCI_ERR_COR_MASK,

                 PCI_ERR_COR_SUPPORTED);



    /* capabilities and control. multiple header logging is supported */

    if (dev->exp.aer_log.log_max > 0) {

        pci_set_long(dev->config + offset + PCI_ERR_CAP,

                     PCI_ERR_CAP_ECRC_GENC | PCI_ERR_CAP_ECRC_CHKC |

                     PCI_ERR_CAP_MHRC);

        pci_set_long(dev->wmask + offset + PCI_ERR_CAP,

                     PCI_ERR_CAP_ECRC_GENE | PCI_ERR_CAP_ECRC_CHKE |

                     PCI_ERR_CAP_MHRE);

    } else {

        pci_set_long(dev->config + offset + PCI_ERR_CAP,

                     PCI_ERR_CAP_ECRC_GENC | PCI_ERR_CAP_ECRC_CHKC);

        pci_set_long(dev->wmask + offset + PCI_ERR_CAP,

                     PCI_ERR_CAP_ECRC_GENE | PCI_ERR_CAP_ECRC_CHKE);

    }



    switch (pcie_cap_get_type(dev)) {

    case PCI_EXP_TYPE_ROOT_PORT:

        /* this case will be set by pcie_aer_root_init() */

        /* fallthrough */

    case PCI_EXP_TYPE_DOWNSTREAM:

    case PCI_EXP_TYPE_UPSTREAM:

        pci_word_test_and_set_mask(dev->wmask + PCI_BRIDGE_CONTROL,

                                   PCI_BRIDGE_CTL_SERR);

        pci_long_test_and_set_mask(dev->w1cmask + PCI_STATUS,

                                   PCI_SEC_STATUS_RCV_SYSTEM_ERROR);

        break;

    default:

        /* nothing */

        break;

    }

    return 0;

}
