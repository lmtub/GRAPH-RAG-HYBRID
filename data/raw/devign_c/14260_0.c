bool timerlistgroup_run_timers(QEMUTimerListGroup *tlg)

{

    QEMUClockType type;

    bool progress = false;

    for (type = 0; type < QEMU_CLOCK_MAX; type++) {

        progress |= timerlist_run_timers(tlg->tl[type]);

    }

    return progress;

}
