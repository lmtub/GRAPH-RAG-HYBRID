static void pci_device_reset(PCIDevice *dev)

{

    int r;



    memset(dev->irq_state, 0, sizeof dev->irq_state);

    dev->config[PCI_COMMAND] &= ~(PCI_COMMAND_IO | PCI_COMMAND_MEMORY |

                                  PCI_COMMAND_MASTER);

    dev->config[PCI_CACHE_LINE_SIZE] = 0x0;

    dev->config[PCI_INTERRUPT_LINE] = 0x0;

    for (r = 0; r < PCI_NUM_REGIONS; ++r) {

        if (!dev->io_regions[r].size) {

            continue;

        }

        pci_set_long(dev->config + pci_bar(dev, r), dev->io_regions[r].type);

    }

    pci_update_mappings(dev);

}
