int armv7m_nvic_acknowledge_irq(void *opaque)

{

    nvic_state *s = (nvic_state *)opaque;

    uint32_t irq;



    irq = gic_acknowledge_irq(&s->gic, 0);

    if (irq == 1023)

        hw_error("Interrupt but no vector\n");

    if (irq >= 32)

        irq -= 16;

    return irq;

}
