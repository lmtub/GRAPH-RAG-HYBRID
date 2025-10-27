void pci_bridge_exitfn(PCIDevice *pci_dev)

{

    PCIBridge *s = DO_UPCAST(PCIBridge, dev, pci_dev);

    assert(QLIST_EMPTY(&s->sec_bus.child));

    QLIST_REMOVE(&s->sec_bus, sibling);

    pci_bridge_region_cleanup(s);

    memory_region_destroy(&s->address_space_mem);

    memory_region_destroy(&s->address_space_io);

    /* qbus_free() is called automatically by qdev_free() */

}
