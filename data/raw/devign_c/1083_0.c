struct omap_mmc_s *omap_mmc_init(hwaddr base,

                MemoryRegion *sysmem,

                BlockDriverState *bd,

                qemu_irq irq, qemu_irq dma[], omap_clk clk)

{

    struct omap_mmc_s *s = (struct omap_mmc_s *)

            g_malloc0(sizeof(struct omap_mmc_s));



    s->irq = irq;

    s->dma = dma;

    s->clk = clk;

    s->lines = 1;	/* TODO: needs to be settable per-board */

    s->rev = 1;



    omap_mmc_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_mmc_ops, s, "omap.mmc", 0x800);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    /* Instantiate the storage */

    s->card = sd_init(bd, false);

    if (s->card == NULL) {

        exit(1);

    }



    return s;

}
