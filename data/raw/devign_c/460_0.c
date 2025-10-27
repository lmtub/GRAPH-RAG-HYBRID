static void dma_init2(struct dma_cont *d, int base, int dshift,

                      int page_base, int pageh_base,

                      qemu_irq *cpu_request_exit)

{

    int i;



    d->dshift = dshift;

    d->cpu_request_exit = cpu_request_exit;



    memory_region_init_io(&d->channel_io, NULL, &channel_io_ops, d,

                          "dma-chan", 8 << d->dshift);

    memory_region_add_subregion(isa_address_space_io(NULL),

                                base, &d->channel_io);



    isa_register_portio_list(NULL, page_base, page_portio_list, d,

                             "dma-page");

    if (pageh_base >= 0) {

        isa_register_portio_list(NULL, pageh_base, pageh_portio_list, d,

                                 "dma-pageh");

    }



    memory_region_init_io(&d->cont_io, NULL, &cont_io_ops, d, "dma-cont",

                          8 << d->dshift);

    memory_region_add_subregion(isa_address_space_io(NULL),

                                base + (8 << d->dshift), &d->cont_io);



    qemu_register_reset(dma_reset, d);

    dma_reset(d);

    for (i = 0; i < ARRAY_SIZE (d->regs); ++i) {

        d->regs[i].transfer_handler = dma_phony_handler;

    }

}
