static void slavio_timer_reset(DeviceState *d)

{

    SLAVIO_TIMERState *s = container_of(d, SLAVIO_TIMERState, busdev.qdev);

    unsigned int i;

    CPUTimerState *curr_timer;



    for (i = 0; i <= MAX_CPUS; i++) {

        curr_timer = &s->cputimer[i];

        curr_timer->limit = 0;

        curr_timer->count = 0;

        curr_timer->reached = 0;

        if (i < s->num_cpus) {

            ptimer_set_limit(curr_timer->timer,

                             LIMIT_TO_PERIODS(TIMER_MAX_COUNT32), 1);

            ptimer_run(curr_timer->timer, 0);

        }

        curr_timer->running = 1;

    }

    s->cputimer_mode = 0;

}
