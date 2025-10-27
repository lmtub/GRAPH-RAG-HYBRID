static void bmdma_writeb(void *opaque, uint32_t addr, uint32_t val)

{

    BMDMAState *bm = opaque;

    PCIIDEState *pci_dev = pci_from_bm(bm);

#ifdef DEBUG_IDE

    printf("bmdma: writeb 0x%02x : 0x%02x\n", addr, val);

#endif

    switch(addr & 3) {

    case 1:

        pci_dev->dev.config[MRDMODE] =

            (pci_dev->dev.config[MRDMODE] & ~0x30) | (val & 0x30);

        cmd646_update_irq(pci_dev);

        break;

    case 2:

        bm->status = (val & 0x60) | (bm->status & 1) | (bm->status & ~val & 0x06);

        break;

    case 3:

        if (bm->unit == 0)

            pci_dev->dev.config[UDIDETCR0] = val;

        else

            pci_dev->dev.config[UDIDETCR1] = val;

        break;

    }

}
