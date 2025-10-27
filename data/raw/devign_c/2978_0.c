void timer_deinit(QEMUTimer *ts)

{

    assert(ts->expire_time == -1);

    ts->timer_list = NULL;

}
