static int64_t cpu_get_clock_locked(void)

{

    int64_t ti;



    if (!timers_state.cpu_ticks_enabled) {

        ti = timers_state.cpu_clock_offset;

    } else {

        ti = get_clock();

        ti += timers_state.cpu_clock_offset;

    }



    return ti;

}
