static bool msix_vector_masked(PCIDevice *dev, int vector, bool fmask)

{

    unsigned offset = vector * PCI_MSIX_ENTRY_SIZE + PCI_MSIX_ENTRY_VECTOR_CTRL;

    return fmask || dev->msix_table[offset] & PCI_MSIX_ENTRY_CTRL_MASKBIT;

}
