static bool timer_mod_ns_locked(QEMUTimerList *timer_list,

                                QEMUTimer *ts, int64_t expire_time)

{

    QEMUTimer **pt, *t;



    /* add the timer in the sorted list */

    pt = &timer_list->active_timers;

    for (;;) {

        t = *pt;

        if (!timer_expired_ns(t, expire_time)) {

            break;

        }

        pt = &t->next;

    }

    ts->expire_time = MAX(expire_time, 0);

    ts->next = *pt;

    *pt = ts;



    return pt == &timer_list->active_timers;

}
