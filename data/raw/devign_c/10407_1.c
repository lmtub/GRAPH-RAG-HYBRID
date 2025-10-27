void cpu_disable_ticks(void)

{

    /* Here, the really thing protected by seqlock is cpu_clock_offset. */

    seqlock_write_lock(&timers_state.vm_clock_seqlock);

    if (timers_state.cpu_ticks_enabled) {

        timers_state.cpu_ticks_offset = cpu_get_ticks();

        timers_state.cpu_clock_offset = cpu_get_clock_locked();

        timers_state.cpu_ticks_enabled = 0;

    }

    seqlock_write_unlock(&timers_state.vm_clock_seqlock);

}
