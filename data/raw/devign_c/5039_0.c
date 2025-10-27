static int nic_can_receive(void *opaque)

{

    dp8393xState *s = opaque;



    if (!(s->regs[SONIC_CR] & SONIC_CR_RXEN))

        return 0;

    if (s->regs[SONIC_ISR] & SONIC_ISR_RBE)

        return 0;

    return 1;

}
