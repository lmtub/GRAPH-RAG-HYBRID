void timer_mod(QEMUTimer *ts, int64_t expire_time)

{

    timer_mod_ns(ts, expire_time * ts->scale);

}
