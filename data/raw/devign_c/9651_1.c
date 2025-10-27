static void pcie_pci_bridge_write_config(PCIDevice *d,

        uint32_t address, uint32_t val, int len)

{

    pci_bridge_write_config(d, address, val, len);

    msi_write_config(d, address, val, len);

    shpc_cap_write_config(d, address, val, len);

}
