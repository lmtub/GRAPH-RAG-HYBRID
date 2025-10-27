bool qemu_clock_run_all_timers(void)

{

    bool progress = false;

    QEMUClockType type;



    for (type = 0; type < QEMU_CLOCK_MAX; type++) {

        progress |= qemu_clock_run_timers(type);

    }



    return progress;

}
