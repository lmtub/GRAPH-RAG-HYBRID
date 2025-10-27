void qemu_clock_enable(QEMUClockType type, bool enabled)

{

    QEMUClock *clock = qemu_clock_ptr(type);

    QEMUTimerList *tl;

    bool old = clock->enabled;

    clock->enabled = enabled;

    if (enabled && !old) {

        qemu_clock_notify(type);

    } else if (!enabled && old) {

        QLIST_FOREACH(tl, &clock->timerlists, list) {

            qemu_event_wait(&tl->timers_done_ev);

        }

    }

}
