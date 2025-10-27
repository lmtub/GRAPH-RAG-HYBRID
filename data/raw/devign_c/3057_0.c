static void init_timers(void)

{

    init_get_clock();

    rt_clock = qemu_new_clock(QEMU_TIMER_REALTIME);

    vm_clock = qemu_new_clock(QEMU_TIMER_VIRTUAL);

}
