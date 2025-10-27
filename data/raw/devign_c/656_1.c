static int pci_bridge_dev_initfn(PCIDevice *dev)

{

    PCIBridge *br = DO_UPCAST(PCIBridge, dev, dev);

    PCIBridgeDev *bridge_dev = DO_UPCAST(PCIBridgeDev, bridge, br);

    int err;

    pci_bridge_map_irq(br, NULL, pci_bridge_dev_map_irq_fn);

    err = pci_bridge_initfn(dev);

    if (err) {

        goto bridge_error;

    }

    memory_region_init(&bridge_dev->bar, "shpc-bar", shpc_bar_size(dev));

    err = shpc_init(dev, &br->sec_bus, &bridge_dev->bar, 0);

    if (err) {

        goto shpc_error;

    }

    err = slotid_cap_init(dev, 0, bridge_dev->chassis_nr, 0);

    if (err) {

        goto slotid_error;

    }

    if ((bridge_dev->flags & (1 << PCI_BRIDGE_DEV_F_MSI_REQ)) &&

        msi_supported) {

        err = msi_init(dev, 0, 1, true, true);

        if (err < 0) {

            goto msi_error;

        }

    }

    /* TODO: spec recommends using 64 bit prefetcheable BAR.

     * Check whether that works well. */

    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY |

		     PCI_BASE_ADDRESS_MEM_TYPE_64, &bridge_dev->bar);

    dev->config[PCI_INTERRUPT_PIN] = 0x1;

    return 0;

msi_error:

    slotid_cap_cleanup(dev);

slotid_error:

    shpc_cleanup(dev, &bridge_dev->bar);

shpc_error:

    memory_region_destroy(&bridge_dev->bar);

bridge_error:

    return err;

}
