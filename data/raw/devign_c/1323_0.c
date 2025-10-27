int64_t av_add_stable(AVRational ts_tb, int64_t ts, AVRational inc_tb, int64_t inc)

{

    inc_tb = av_mul_q(inc_tb, (AVRational) {inc, 1});



    if (av_cmp_q(inc_tb, ts_tb) < 0) {

        //increase step is too small for even 1 step to be representable

        return ts;

    } else {

        int64_t old = av_rescale_q(ts, ts_tb, inc_tb);

        int64_t old_ts = av_rescale_q(old, inc_tb, ts_tb);

        return av_rescale_q(old + 1, inc_tb, ts_tb) + (ts - old_ts);

    }

}
