bool qemu_clock_expired(QEMUClockType type)

{

    return timerlist_expired(

        main_loop_tlg.tl[type]);

}
