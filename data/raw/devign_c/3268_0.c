static void msix_set_pending(PCIDevice *dev, int vector)

{

    *msix_pending_byte(dev, vector) |= msix_pending_mask(vector);

}
