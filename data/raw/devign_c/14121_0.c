PXA2xxMMCIState *pxa2xx_mmci_init(MemoryRegion *sysmem,

                hwaddr base,

                BlockDriverState *bd, qemu_irq irq,

                qemu_irq rx_dma, qemu_irq tx_dma)

{

    PXA2xxMMCIState *s;



    s = (PXA2xxMMCIState *) g_malloc0(sizeof(PXA2xxMMCIState));

    s->irq = irq;

    s->rx_dma = rx_dma;

    s->tx_dma = tx_dma;



    memory_region_init_io(&s->iomem, NULL, &pxa2xx_mmci_ops, s,

                          "pxa2xx-mmci", 0x00100000);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    /* Instantiate the actual storage */

    s->card = sd_init(bd, false);

    if (s->card == NULL) {

        exit(1);

    }



    register_savevm(NULL, "pxa2xx_mmci", 0, 0,

                    pxa2xx_mmci_save, pxa2xx_mmci_load, s);



    return s;

}
