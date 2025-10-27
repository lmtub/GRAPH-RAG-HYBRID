static void __attribute__((constructor)) init_main_loop(void)

{

    init_clocks();

    init_timer_alarm();

    qemu_clock_enable(vm_clock, false);

}
