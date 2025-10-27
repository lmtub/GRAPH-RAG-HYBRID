void timerlist_free(QEMUTimerList *timer_list)

{

    assert(!timerlist_has_timers(timer_list));

    if (timer_list->clock) {

        QLIST_REMOVE(timer_list, list);

    }

    qemu_mutex_destroy(&timer_list->active_timers_lock);

    g_free(timer_list);

}
