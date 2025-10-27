void pci_bridge_initfn(PCIDevice *dev, const char *typename)

{

    PCIBus *parent = dev->bus;

    PCIBridge *br = PCI_BRIDGE(dev);

    PCIBus *sec_bus = &br->sec_bus;



    pci_word_test_and_set_mask(dev->config + PCI_STATUS,

                               PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);



    /*

     * TODO: We implement VGA Enable in the Bridge Control Register

     * therefore per the PCI to PCI bridge spec we must also implement

     * VGA Palette Snooping.  When done, set this bit writable:

     *

     * pci_word_test_and_set_mask(dev->wmask + PCI_COMMAND,

     *                            PCI_COMMAND_VGA_PALETTE);

     */



    pci_config_set_class(dev->config, PCI_CLASS_BRIDGE_PCI);

    dev->config[PCI_HEADER_TYPE] =

        (dev->config[PCI_HEADER_TYPE] & PCI_HEADER_TYPE_MULTI_FUNCTION) |

        PCI_HEADER_TYPE_BRIDGE;

    pci_set_word(dev->config + PCI_SEC_STATUS,

                 PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);



    /*

     * If we don't specify the name, the bus will be addressed as <id>.0, where

     * id is the device id.

     * Since PCI Bridge devices have a single bus each, we don't need the index:

     * let users address the bus using the device name.

     */

    if (!br->bus_name && dev->qdev.id && *dev->qdev.id) {

	    br->bus_name = dev->qdev.id;

    }



    qbus_create_inplace(sec_bus, sizeof(br->sec_bus), typename, DEVICE(dev),

                        br->bus_name);

    sec_bus->parent_dev = dev;

    sec_bus->map_irq = br->map_irq ? br->map_irq : pci_swizzle_map_irq_fn;

    sec_bus->address_space_mem = &br->address_space_mem;

    memory_region_init(&br->address_space_mem, OBJECT(br), "pci_bridge_pci", UINT64_MAX);

    sec_bus->address_space_io = &br->address_space_io;

    memory_region_init(&br->address_space_io, OBJECT(br), "pci_bridge_io", 65536);

    br->windows = pci_bridge_region_init(br);

    QLIST_INIT(&sec_bus->child);

    QLIST_INSERT_HEAD(&parent->child, sec_bus, sibling);

}
