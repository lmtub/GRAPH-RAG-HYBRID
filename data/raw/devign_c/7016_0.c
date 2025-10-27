QEMUTimer *qemu_new_timer(QEMUClock *clock, int scale,

                          QEMUTimerCB *cb, void *opaque)

{

    return g_malloc(1);

}
