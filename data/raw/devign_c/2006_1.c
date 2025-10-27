int pcie_cap_v1_init(PCIDevice *dev, uint8_t offset, uint8_t type,

                     uint8_t port)

{

    /* PCIe cap v1 init */

    int pos;

    uint8_t *exp_cap;



    assert(pci_is_express(dev));



    pos = pci_add_capability(dev, PCI_CAP_ID_EXP, offset, PCI_EXP_VER1_SIZEOF);

    if (pos < 0) {

        return pos;

    }

    dev->exp.exp_cap = pos;

    exp_cap = dev->config + pos;



    pcie_cap_v1_fill(exp_cap, port, type, PCI_EXP_FLAGS_VER1);



    return pos;

}
