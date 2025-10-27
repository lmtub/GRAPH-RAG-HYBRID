int pcie_cap_init(PCIDevice *dev, uint8_t offset, uint8_t type, uint8_t port)

{

    /* PCIe cap v2 init */

    int pos;

    uint8_t *exp_cap;



    assert(pci_is_express(dev));



    pos = pci_add_capability(dev, PCI_CAP_ID_EXP, offset, PCI_EXP_VER2_SIZEOF);

    if (pos < 0) {

        return pos;

    }

    dev->exp.exp_cap = pos;

    exp_cap = dev->config + pos;



    /* Filling values common with v1 */

    pcie_cap_v1_fill(exp_cap, port, type, PCI_EXP_FLAGS_VER2);



    /* Filling v2 specific values */

    pci_set_long(exp_cap + PCI_EXP_DEVCAP2,

                 PCI_EXP_DEVCAP2_EFF | PCI_EXP_DEVCAP2_EETLPP);



    pci_set_word(dev->wmask + pos + PCI_EXP_DEVCTL2, PCI_EXP_DEVCTL2_EETLPPB);

    return pos;

}
