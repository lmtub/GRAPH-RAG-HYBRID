static void ratelimit_set_speed(RateLimit *limit, uint64_t speed)

{

    limit->slice_quota = speed / (1000000000ULL / SLICE_TIME);

}
