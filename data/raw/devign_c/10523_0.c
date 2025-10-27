static bool timer_expired_ns(QEMUTimer *timer_head, int64_t current_time)

{

    return timer_head && (timer_head->expire_time <= current_time);

}
