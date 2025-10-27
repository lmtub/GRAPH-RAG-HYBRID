static int pcibus_reset(BusState *qbus)

{

    pci_bus_reset(DO_UPCAST(PCIBus, qbus, qbus));



    /* topology traverse is done by pci_bus_reset().

       Tell qbus/qdev walker not to traverse the tree */

    return 1;

}
