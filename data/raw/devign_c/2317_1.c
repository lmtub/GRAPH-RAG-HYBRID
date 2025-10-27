static void test_leak_bucket(void)
{
    throttle_config_init(&cfg);
    bkt = cfg.buckets[THROTTLE_BPS_TOTAL];
    /* set initial value */
    bkt.avg = 150;
    bkt.max = 15;
    bkt.level = 1.5;
    /* leak an op work of time */
    throttle_leak_bucket(&bkt, NANOSECONDS_PER_SECOND / 150);
    g_assert(bkt.avg == 150);
    g_assert(bkt.max == 15);
    g_assert(double_cmp(bkt.level, 0.5));
    /* leak again emptying the bucket */
    throttle_leak_bucket(&bkt, NANOSECONDS_PER_SECOND / 150);
    g_assert(bkt.avg == 150);
    g_assert(bkt.max == 15);
    g_assert(double_cmp(bkt.level, 0));
    /* check that the bucket level won't go lower */
    throttle_leak_bucket(&bkt, NANOSECONDS_PER_SECOND / 150);
    g_assert(bkt.avg == 150);
    g_assert(bkt.max == 15);
    g_assert(double_cmp(bkt.level, 0));
}