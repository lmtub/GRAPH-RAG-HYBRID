void pxa25x_timer_init(target_phys_addr_t base, qemu_irq *irqs)

{

    pxa2xx_timer_info *s = pxa2xx_timer_init(base, irqs);

    s->freq = PXA25X_FREQ;

    s->tm4 = NULL;

}
