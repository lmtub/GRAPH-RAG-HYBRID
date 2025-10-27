struct omap_uart_s *omap_uart_init(hwaddr base,

                qemu_irq irq, omap_clk fclk, omap_clk iclk,

                qemu_irq txdma, qemu_irq rxdma,

                const char *label, CharDriverState *chr)

{

    struct omap_uart_s *s = (struct omap_uart_s *)

            g_malloc0(sizeof(struct omap_uart_s));



    s->base = base;

    s->fclk = fclk;

    s->irq = irq;

    s->serial = serial_mm_init(get_system_memory(), base, 2, irq,

                               omap_clk_getrate(fclk)/16,

                               chr ?: qemu_chr_new(label, "null", NULL),

                               DEVICE_NATIVE_ENDIAN);

    return s;

}
