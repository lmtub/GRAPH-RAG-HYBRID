QEMUTimerList *timerlist_new(QEMUClockType type,

                             QEMUTimerListNotifyCB *cb,

                             void *opaque)

{

    QEMUTimerList *timer_list;

    QEMUClock *clock = qemu_clock_ptr(type);



    timer_list = g_malloc0(sizeof(QEMUTimerList));

    qemu_event_init(&timer_list->timers_done_ev, true);

    timer_list->clock = clock;

    timer_list->notify_cb = cb;

    timer_list->notify_opaque = opaque;

    qemu_mutex_init(&timer_list->active_timers_lock);

    QLIST_INSERT_HEAD(&clock->timerlists, timer_list, list);

    return timer_list;

}
