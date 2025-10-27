static int64_t ratelimit_calculate_delay(RateLimit *limit, uint64_t n)

{

    int64_t delay_ns = 0;

    int64_t now = qemu_get_clock_ns(rt_clock);



    if (limit->next_slice_time < now) {

        limit->next_slice_time = now + SLICE_TIME;

        limit->dispatched = 0;

    }

    if (limit->dispatched + n > limit->slice_quota) {

        delay_ns = limit->next_slice_time - now;

    } else {

        limit->dispatched += n;

    }

    return delay_ns;

}
