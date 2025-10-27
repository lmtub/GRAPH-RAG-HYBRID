uint64_t timer_expire_time_ns(QEMUTimer *ts)

{

    return timer_pending(ts) ? ts->expire_time : -1;

}
