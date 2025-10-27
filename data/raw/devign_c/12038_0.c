static void win32_rearm_timer(struct qemu_alarm_timer *t)

{

    struct qemu_alarm_win32 *data = t->priv;

    uint64_t nearest_delta_us;



    if (!active_timers[QEMU_TIMER_REALTIME] &&

                !active_timers[QEMU_TIMER_VIRTUAL])

        return;



    nearest_delta_us = qemu_next_deadline_dyntick();

    nearest_delta_us /= 1000;



    timeKillEvent(data->timerId);



    data->timerId = timeSetEvent(1,

                        data->period,

                        host_alarm_handler,

                        (DWORD)t,

                        TIME_ONESHOT | TIME_PERIODIC);



    if (!data->timerId) {

        fprintf(stderr, "Failed to re-arm win32 alarm timer %ld\n",

                GetLastError());



        timeEndPeriod(data->period);

        exit(1);

    }

}
