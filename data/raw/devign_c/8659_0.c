inline qemu_irq omap_inth_get_pin(struct omap_intr_handler_s *s, int n)

{

    return s->pins[n];

}
