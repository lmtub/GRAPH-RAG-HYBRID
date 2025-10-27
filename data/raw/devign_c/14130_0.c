int64_t timerlist_deadline_ns(QEMUTimerList *timer_list)

{

    int64_t delta;



    if (!timer_list->clock->enabled || !timer_list->active_timers) {

        return -1;

    }



    delta = timer_list->active_timers->expire_time -

        qemu_clock_get_ns(timer_list->clock->type);



    if (delta <= 0) {

        return 0;

    }



    return delta;

}
