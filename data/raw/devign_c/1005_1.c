bool throttle_is_valid(ThrottleConfig *cfg, Error **errp)
{
    int i;
    bool bps_flag, ops_flag;
    bool bps_max_flag, ops_max_flag;
    bps_flag = cfg->buckets[THROTTLE_BPS_TOTAL].avg &&
               (cfg->buckets[THROTTLE_BPS_READ].avg ||
                cfg->buckets[THROTTLE_BPS_WRITE].avg);
    ops_flag = cfg->buckets[THROTTLE_OPS_TOTAL].avg &&
               (cfg->buckets[THROTTLE_OPS_READ].avg ||
                cfg->buckets[THROTTLE_OPS_WRITE].avg);
    bps_max_flag = cfg->buckets[THROTTLE_BPS_TOTAL].max &&
                  (cfg->buckets[THROTTLE_BPS_READ].max  ||
                   cfg->buckets[THROTTLE_BPS_WRITE].max);
    ops_max_flag = cfg->buckets[THROTTLE_OPS_TOTAL].max &&
                   (cfg->buckets[THROTTLE_OPS_READ].max ||
                   cfg->buckets[THROTTLE_OPS_WRITE].max);
    if (bps_flag || ops_flag || bps_max_flag || ops_max_flag) {
        error_setg(errp, "bps/iops/max total values and read/write values"
                   " cannot be used at the same time");
    if (cfg->op_size &&
        !cfg->buckets[THROTTLE_OPS_TOTAL].avg &&
        !cfg->buckets[THROTTLE_OPS_READ].avg &&
        !cfg->buckets[THROTTLE_OPS_WRITE].avg) {
        error_setg(errp, "iops size requires an iops value to be set");
    for (i = 0; i < BUCKETS_COUNT; i++) {
        LeakyBucket *bkt = &cfg->buckets[i];
        if (bkt->avg > THROTTLE_VALUE_MAX || bkt->max > THROTTLE_VALUE_MAX) {
            error_setg(errp, "bps/iops/max values must be within [0, %lld]",
                       THROTTLE_VALUE_MAX);
        if (!bkt->burst_length) {
            error_setg(errp, "the burst length cannot be 0");
        if (bkt->burst_length > 1 && !bkt->max) {
            error_setg(errp, "burst length set without burst rate");
        if (bkt->max && !bkt->avg) {
            error_setg(errp, "bps_max/iops_max require corresponding"
                       " bps/iops values");
        if (bkt->max && bkt->max < bkt->avg) {
            error_setg(errp, "bps_max/iops_max cannot be lower than bps/iops");
    return true;