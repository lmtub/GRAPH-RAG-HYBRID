qemu_irq sh7750_irl(SH7750State *s)

{

    sh_intc_toggle_source(sh_intc_source(&s->intc, IRL), 1, 0); /* enable */

    return qemu_allocate_irqs(sh_intc_set_irl, sh_intc_source(&s->intc, IRL),

                               1)[0];

}
