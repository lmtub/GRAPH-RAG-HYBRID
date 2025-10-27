static void rc4030_reset(DeviceState *dev)

{

    rc4030State *s = RC4030(dev);

    int i;



    s->config = 0x410; /* some boards seem to accept 0x104 too */

    s->revision = 1;

    s->invalid_address_register = 0;



    memset(s->dma_regs, 0, sizeof(s->dma_regs));

    rc4030_dma_tt_update(s, 0, 0);



    s->remote_failed_address = s->memory_failed_address = 0;

    s->cache_maint = 0;

    s->cache_ptag = s->cache_ltag = 0;

    s->cache_bmask = 0;



    s->memory_refresh_rate = 0x18186;

    s->nvram_protect = 7;

    for (i = 0; i < 15; i++)

        s->rem_speed[i] = 7;

    s->imr_jazz = 0x10; /* XXX: required by firmware, but why? */

    s->isr_jazz = 0;



    s->itr = 0;



    qemu_irq_lower(s->timer_irq);

    qemu_irq_lower(s->jazz_bus_irq);

}
