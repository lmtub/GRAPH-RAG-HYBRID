static uint32_t nam_readw (void *opaque, uint32_t addr)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    uint32_t val = ~0U;

    uint32_t index = addr - s->base[0];

    s->cas = 0;

    val = mixer_load (s, index);

    return val;

}
