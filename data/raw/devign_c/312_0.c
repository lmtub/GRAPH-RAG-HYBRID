static void icount_warp_rt(void)

{

    unsigned seq;

    int64_t warp_start;



    /* The icount_warp_timer is rescheduled soon after vm_clock_warp_start

     * changes from -1 to another value, so the race here is okay.

     */

    do {

        seq = seqlock_read_begin(&timers_state.vm_clock_seqlock);

        warp_start = vm_clock_warp_start;

    } while (seqlock_read_retry(&timers_state.vm_clock_seqlock, seq));



    if (warp_start == -1) {

        return;

    }



    seqlock_write_begin(&timers_state.vm_clock_seqlock);

    if (runstate_is_running()) {

        int64_t clock = REPLAY_CLOCK(REPLAY_CLOCK_VIRTUAL_RT,

                                     cpu_get_clock_locked());

        int64_t warp_delta;



        warp_delta = clock - vm_clock_warp_start;

        if (use_icount == 2) {

            /*

             * In adaptive mode, do not let QEMU_CLOCK_VIRTUAL run too

             * far ahead of real time.

             */

            int64_t cur_icount = cpu_get_icount_locked();

            int64_t delta = clock - cur_icount;

            warp_delta = MIN(warp_delta, delta);

        }

        timers_state.qemu_icount_bias += warp_delta;

    }

    vm_clock_warp_start = -1;

    seqlock_write_end(&timers_state.vm_clock_seqlock);



    if (qemu_clock_expired(QEMU_CLOCK_VIRTUAL)) {

        qemu_clock_notify(QEMU_CLOCK_VIRTUAL);

    }

}
