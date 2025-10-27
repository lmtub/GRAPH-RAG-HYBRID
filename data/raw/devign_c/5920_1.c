void throttle_account(ThrottleState *ts, bool is_write, uint64_t size)

{

    double units = 1.0;



    /* if cfg.op_size is defined and smaller than size we compute unit count */

    if (ts->cfg.op_size && size > ts->cfg.op_size) {

        units = (double) size / ts->cfg.op_size;

    }



    ts->cfg.buckets[THROTTLE_BPS_TOTAL].level += size;

    ts->cfg.buckets[THROTTLE_OPS_TOTAL].level += units;



    if (is_write) {

        ts->cfg.buckets[THROTTLE_BPS_WRITE].level += size;

        ts->cfg.buckets[THROTTLE_OPS_WRITE].level += units;

    } else {

        ts->cfg.buckets[THROTTLE_BPS_READ].level += size;

        ts->cfg.buckets[THROTTLE_OPS_READ].level += units;

    }

}
