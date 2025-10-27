void bmdma_init(IDEBus *bus, BMDMAState *bm, PCIIDEState *d)

{

    qemu_irq *irq;



    if (bus->dma == &bm->dma) {

        return;

    }



    bm->dma.ops = &bmdma_ops;

    bus->dma = &bm->dma;

    bm->irq = bus->irq;

    irq = qemu_allocate_irqs(bmdma_irq, bm, 1);

    bus->irq = *irq;

    bm->pci_dev = d;

}
