bool timer_pending(QEMUTimer *ts)

{

    QEMUTimer *t;

    for (t = ts->timer_list->active_timers; t != NULL; t = t->next) {

        if (t == ts) {

            return true;

        }

    }

    return false;

}
