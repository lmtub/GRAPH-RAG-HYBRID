QEMUTimer *qemu_new_timer(QEMUClock *clock, QEMUTimerCB *cb, void *opaque)

{

    QEMUTimer *ts;



    ts = qemu_mallocz(sizeof(QEMUTimer));

    ts->clock = clock;

    ts->cb = cb;

    ts->opaque = opaque;

    return ts;

}
