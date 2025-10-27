bool qemu_clock_run_timers(QEMUClockType type)

{

    return timerlist_run_timers(main_loop_tlg.tl[type]);

}
