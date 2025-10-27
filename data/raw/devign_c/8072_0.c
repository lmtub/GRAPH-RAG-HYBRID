static void mch_update_pciexbar(MCHPCIState *mch)

{

    PCIDevice *pci_dev = PCI_DEVICE(mch);

    BusState *bus = qdev_get_parent_bus(DEVICE(mch));

    PCIExpressHost *pehb = PCIE_HOST_BRIDGE(bus->parent);



    uint64_t pciexbar;

    int enable;

    uint64_t addr;

    uint64_t addr_mask;

    uint32_t length;



    pciexbar = pci_get_quad(pci_dev->config + MCH_HOST_BRIDGE_PCIEXBAR);

    enable = pciexbar & MCH_HOST_BRIDGE_PCIEXBAREN;

    addr_mask = MCH_HOST_BRIDGE_PCIEXBAR_ADMSK;

    switch (pciexbar & MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_MASK) {

    case MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_256M:

        length = 256 * 1024 * 1024;

        break;

    case MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_128M:

        length = 128 * 1024 * 1024;

        addr_mask |= MCH_HOST_BRIDGE_PCIEXBAR_128ADMSK |

            MCH_HOST_BRIDGE_PCIEXBAR_64ADMSK;

        break;

    case MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_64M:

        length = 64 * 1024 * 1024;

        addr_mask |= MCH_HOST_BRIDGE_PCIEXBAR_64ADMSK;

        break;

    case MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_RVD:

    default:

        enable = 0;

        length = 0;

        abort();

        break;

    }

    addr = pciexbar & addr_mask;

    pcie_host_mmcfg_update(pehb, enable, addr, length);

    /* Leave enough space for the MCFG BAR */

    /*

     * TODO: this matches current bios behaviour, but it's not a power of two,

     * which means an MTRR can't cover it exactly.

     */

    if (enable) {

        mch->pci_hole.begin = addr + length;

    } else {

        mch->pci_hole.begin = MCH_HOST_BRIDGE_PCIEXBAR_DEFAULT;

    }

}
