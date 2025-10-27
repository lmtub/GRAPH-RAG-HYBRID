bool qemu_clock_has_timers(QEMUClockType type)

{

    return timerlist_has_timers(

        main_loop_tlg.tl[type]);

}
