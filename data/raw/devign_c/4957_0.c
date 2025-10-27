void timerlistgroup_deinit(QEMUTimerListGroup *tlg)

{

    QEMUClockType type;

    for (type = 0; type < QEMU_CLOCK_MAX; type++) {

        timerlist_free(tlg->tl[type]);

    }

}
