static int rtc_post_load(void *opaque, int version_id)

{

    RTCState *s = opaque;



    if (version_id <= 2 || rtc_clock == QEMU_CLOCK_REALTIME) {

        rtc_set_time(s);

        s->offset = 0;

        check_update_timer(s);

    }



    uint64_t now = qemu_clock_get_ns(rtc_clock);

    if (now < s->next_periodic_time ||

        now > (s->next_periodic_time + get_max_clock_jump())) {

        periodic_timer_update(s, qemu_clock_get_ns(rtc_clock));

    }



#ifdef TARGET_I386

    if (version_id >= 2) {

        if (s->lost_tick_policy == LOST_TICK_POLICY_SLEW) {

            rtc_coalesced_timer_update(s);

        }

    }

#endif

    return 0;

}
