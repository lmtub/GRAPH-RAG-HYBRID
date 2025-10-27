uint64_t cpu_tick_get_count(CPUTimer *timer)

{

    uint64_t real_count = timer_to_cpu_ticks(

                    qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) - timer->clock_offset,

                    timer->frequency);



    TIMER_DPRINTF("%s get_count count=0x%016lx (%s) p=%p\n",

           timer->name, real_count,

           timer->disabled?"disabled":"enabled", timer);



    if (timer->disabled)

        real_count |= timer->disabled_mask;



    return real_count;

}
