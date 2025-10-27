void cpu_tick_set_count(CPUTimer *timer, uint64_t count)

{

    uint64_t real_count = count & ~timer->disabled_mask;

    uint64_t disabled_bit = count & timer->disabled_mask;



    int64_t vm_clock_offset = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) -

                    cpu_to_timer_ticks(real_count, timer->frequency);



    TIMER_DPRINTF("%s set_count count=0x%016lx (%s) p=%p\n",

                  timer->name, real_count,

                  timer->disabled?"disabled":"enabled", timer);



    timer->disabled = disabled_bit ? 1 : 0;

    timer->clock_offset = vm_clock_offset;

}
