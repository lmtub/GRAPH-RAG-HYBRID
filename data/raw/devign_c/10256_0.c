static void timer_start(SpiceTimer *timer, uint32_t ms)

{

    qemu_mod_timer(timer->timer, qemu_get_clock(rt_clock) + ms);

}
