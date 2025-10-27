static void pcie_pci_bridge_reset(DeviceState *qdev)

{

    PCIDevice *d = PCI_DEVICE(qdev);

    pci_bridge_reset(qdev);

    msi_reset(d);

    shpc_reset(d);

}
