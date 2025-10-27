static void set_next_tick(rc4030State *s)

{

    qemu_irq_lower(s->timer_irq);

    uint32_t hz;



    hz = 1000 / (s->itr + 1);



    qemu_mod_timer(s->periodic_timer, qemu_get_clock(vm_clock) + ticks_per_sec / hz);

}
