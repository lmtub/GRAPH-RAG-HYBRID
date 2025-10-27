struct omap_intr_handler_s *omap2_inth_init(target_phys_addr_t base,

                int size, int nbanks, qemu_irq **pins,

                qemu_irq parent_irq, qemu_irq parent_fiq,

                omap_clk fclk, omap_clk iclk)

{

    struct omap_intr_handler_s *s = (struct omap_intr_handler_s *)

            g_malloc0(sizeof(struct omap_intr_handler_s) +

                            sizeof(struct omap_intr_handler_bank_s) * nbanks);



    s->parent_intr[0] = parent_irq;

    s->parent_intr[1] = parent_fiq;

    s->nbanks = nbanks;

    s->level_only = 1;

    s->pins = qemu_allocate_irqs(omap_set_intr_noedge, s, nbanks * 32);

    if (pins)

        *pins = s->pins;



    memory_region_init_io(&s->mmio, &omap2_inth_mem_ops, s, "omap2-intc", size);

    memory_region_add_subregion(get_system_memory(), base, &s->mmio);



    omap_inth_reset(s);



    return s;

}
