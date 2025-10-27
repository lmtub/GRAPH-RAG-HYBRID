static int smc91c111_can_receive(void *opaque)

{

    smc91c111_state *s = (smc91c111_state *)opaque;



    if ((s->rcr & RCR_RXEN) == 0 || (s->rcr & RCR_SOFT_RST))

        return 1;

    if (s->allocated == (1 << NUM_PACKETS) - 1)

        return 0;

    return 1;

}
