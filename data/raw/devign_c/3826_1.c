void timer_del(QEMUTimer *ts)

{

    QEMUTimerList *timer_list = ts->timer_list;



    qemu_mutex_lock(&timer_list->active_timers_lock);

    timer_del_locked(timer_list, ts);

    qemu_mutex_unlock(&timer_list->active_timers_lock);

}
