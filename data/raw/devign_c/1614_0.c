static void mipsnet_receive(void *opaque, const uint8_t *buf, size_t size)

{

    MIPSnetState *s = opaque;



#ifdef DEBUG_MIPSNET_RECEIVE

    printf("mipsnet: receiving len=%d\n", size);

#endif

    if (!mipsnet_can_receive(opaque))

        return;



    s->busy = 1;



    /* Just accept everything. */



    /* Write packet data. */

    memcpy(s->rx_buffer, buf, size);



    s->rx_count = size;

    s->rx_read = 0;



    /* Now we can signal we have received something. */

    s->intctl |= MIPSNET_INTCTL_RXDONE;

    mipsnet_update_irq(s);

}
