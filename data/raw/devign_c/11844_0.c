struct omap_uart_s *omap2_uart_init(struct omap_target_agent_s *ta,

                qemu_irq irq, omap_clk fclk, omap_clk iclk,

                qemu_irq txdma, qemu_irq rxdma, CharDriverState *chr)

{

    target_phys_addr_t base = omap_l4_attach(ta, 0, 0);

    struct omap_uart_s *s = omap_uart_init(base, irq,

                    fclk, iclk, txdma, rxdma, chr);

    int iomemtype = cpu_register_io_memory(0, omap_uart_readfn,

                    omap_uart_writefn, s);



    s->ta = ta;

    s->base = base;



    cpu_register_physical_memory(s->base + 0x20, 0x100, iomemtype);



    return s;

}
