pcie_cap_v1_fill(uint8_t *exp_cap, uint8_t port, uint8_t type, uint8_t version)

{

    /* capability register

    interrupt message number defaults to 0 */

    pci_set_word(exp_cap + PCI_EXP_FLAGS,

                 ((type << PCI_EXP_FLAGS_TYPE_SHIFT) & PCI_EXP_FLAGS_TYPE) |

                 version);



    /* device capability register

     * table 7-12:

     * roll based error reporting bit must be set by all

     * Functions conforming to the ECN, PCI Express Base

     * Specification, Revision 1.1., or subsequent PCI Express Base

     * Specification revisions.

     */

    pci_set_long(exp_cap + PCI_EXP_DEVCAP, PCI_EXP_DEVCAP_RBER);



    pci_set_long(exp_cap + PCI_EXP_LNKCAP,

                 (port << PCI_EXP_LNKCAP_PN_SHIFT) |

                 PCI_EXP_LNKCAP_ASPMS_0S |

                 PCI_EXP_LNK_MLW_1 |

                 PCI_EXP_LNK_LS_25);



    pci_set_word(exp_cap + PCI_EXP_LNKSTA,

                 PCI_EXP_LNK_MLW_1 | PCI_EXP_LNK_LS_25 |PCI_EXP_LNKSTA_DLLLA);

}
