static void qemu_run_timers(QEMUClock *clock)

{

    QEMUTimer **ptimer_head, *ts;

    int64_t current_time;

   

    if (!clock->enabled)

        return;



    current_time = qemu_get_clock (clock);

    ptimer_head = &active_timers[clock->type];

    for(;;) {

        ts = *ptimer_head;

        if (!ts || ts->expire_time > current_time)

            break;

        /* remove timer from the list before calling the callback */

        *ptimer_head = ts->next;

        ts->next = NULL;



        /* run the callback (the timer list can be modified) */

        ts->cb(ts->opaque);

    }

}
