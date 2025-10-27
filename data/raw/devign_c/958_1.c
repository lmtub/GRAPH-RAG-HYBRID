int64_t cpu_get_clock(void)

{

    int64_t ti;

    if (!timers_state.cpu_ticks_enabled) {

        return timers_state.cpu_clock_offset;

    } else {

        ti = get_clock();

        return ti + timers_state.cpu_clock_offset;

    }

}
