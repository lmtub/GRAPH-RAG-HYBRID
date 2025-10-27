int64_t swr_next_pts(struct SwrContext *s, int64_t pts){

    if(pts == INT64_MIN)

        return s->outpts;

    if(s->min_compensation >= FLT_MAX) {

        return (s->outpts = pts - swr_get_delay(s, s->in_sample_rate * (int64_t)s->out_sample_rate));

    } else {

        int64_t delta = pts - swr_get_delay(s, s->in_sample_rate * (int64_t)s->out_sample_rate) - s->outpts;

        double fdelta = delta /(double)(s->in_sample_rate * (int64_t)s->out_sample_rate);



        if(fabs(fdelta) > s->min_compensation) {

            if(!s->outpts || fabs(fdelta) > s->min_hard_compensation){

                int ret;

                if(delta > 0) ret = swr_inject_silence(s,  delta / s->out_sample_rate);

                else          ret = swr_drop_output   (s, -delta / s-> in_sample_rate);

                if(ret<0){

                    av_log(s, AV_LOG_ERROR, "Failed to compensate for timestamp delta of %f\n", fdelta);

                }

            } else if(s->soft_compensation_duration && s->max_soft_compensation) {

                int duration = s->out_sample_rate * s->soft_compensation_duration;

                double max_soft_compensation = s->max_soft_compensation / (s->max_soft_compensation < 0 ? -s->in_sample_rate : 1);

                int comp = av_clipf(fdelta, -max_soft_compensation, max_soft_compensation) * duration ;

                av_log(s, AV_LOG_VERBOSE, "compensating audio timestamp drift:%f compensation:%d in:%d\n", fdelta, comp, duration);

                swr_set_compensation(s, comp, duration);

            }

        }



        return s->outpts;

    }

}
