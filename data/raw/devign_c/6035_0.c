static void nam_writel (void *opaque, uint32_t addr, uint32_t val)

{

    PCIAC97LinkState *d = opaque;

    AC97LinkState *s = &d->ac97;

    dolog ("U nam writel %#x <- %#x\n", addr, val);

    s->cas = 0;

}
