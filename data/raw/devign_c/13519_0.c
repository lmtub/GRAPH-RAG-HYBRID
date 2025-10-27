static int pxb_map_irq_fn(PCIDevice *pci_dev, int pin)

{

    PCIDevice *pxb = pci_dev->bus->parent_dev;



    /*

     * The bios does not index the pxb slot number when

     * it computes the IRQ because it resides on bus 0

     * and not on the current bus.

     * However QEMU routes the irq through bus 0 and adds

     * the pxb slot to the IRQ computation of the PXB

     * device.

     *

     * Synchronize between bios and QEMU by canceling

     * pxb's effect.

     */

    return pin - PCI_SLOT(pxb->devfn);

}
