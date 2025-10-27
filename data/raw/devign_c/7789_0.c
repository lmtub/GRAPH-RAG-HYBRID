int main_loop_wait(int nonblocking)

{

    int ret;

    uint32_t timeout = UINT32_MAX;

    int64_t timeout_ns;



    if (nonblocking) {

        timeout = 0;

    }



    /* poll any events */

    g_array_set_size(gpollfds, 0); /* reset for new iteration */

    /* XXX: separate device handlers from system ones */

#ifdef CONFIG_SLIRP

    slirp_pollfds_fill(gpollfds, &timeout);

#endif



    if (timeout == UINT32_MAX) {

        timeout_ns = -1;

    } else {

        timeout_ns = (uint64_t)timeout * (int64_t)(SCALE_MS);

    }



    timeout_ns = qemu_soonest_timeout(timeout_ns,

                                      timerlistgroup_deadline_ns(

                                          &main_loop_tlg));



    ret = os_host_main_loop_wait(timeout_ns);

#ifdef CONFIG_SLIRP

    slirp_pollfds_poll(gpollfds, (ret < 0));

#endif



    /* CPU thread can infinitely wait for event after

       missing the warp */

    qemu_start_warp_timer();

    qemu_clock_run_all_timers();



    return ret;

}
