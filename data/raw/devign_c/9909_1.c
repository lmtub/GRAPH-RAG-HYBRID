static int filter_frame(AVFilterLink *inlink, AVFrame *insamples)

{

    AVFilterContext *ctx = inlink->dst;

    AVFilterLink *outlink = ctx->outputs[0];

    ShowCQTContext *s = ctx->priv;

    int remaining, step, ret, x, i, j, m;

    float *audio_data;

    AVFrame *out = NULL;



    if (!insamples) {

        while (s->remaining_fill < s->fft_len / 2) {

            memset(&s->fft_data[s->fft_len - s->remaining_fill], 0, sizeof(*s->fft_data) * s->remaining_fill);

            ret = plot_cqt(ctx, &out);

            if (ret < 0)

                return ret;



            step = s->step + (s->step_frac.num + s->remaining_frac) / s->step_frac.den;

            s->remaining_frac = (s->step_frac.num + s->remaining_frac) % s->step_frac.den;

            for (x = 0; x < (s->fft_len-step); x++)

                s->fft_data[x] = s->fft_data[x+step];

            s->remaining_fill += step;



            if (out)

                return ff_filter_frame(outlink, out);

        }

        return AVERROR_EOF;

    }



    remaining = insamples->nb_samples;

    audio_data = (float*) insamples->data[0];



    while (remaining) {

        i = insamples->nb_samples - remaining;

        j = s->fft_len - s->remaining_fill;

        if (remaining >= s->remaining_fill) {

            for (m = 0; m < s->remaining_fill; m++) {

                s->fft_data[j+m].re = audio_data[2*(i+m)];

                s->fft_data[j+m].im = audio_data[2*(i+m)+1];

            }

            ret = plot_cqt(ctx, &out);

            if (ret < 0) {

                av_frame_free(&insamples);

                return ret;

            }

            remaining -= s->remaining_fill;

            if (out) {

                int64_t pts = av_rescale_q(insamples->pts, inlink->time_base, av_make_q(1, inlink->sample_rate));

                pts += insamples->nb_samples - remaining - s->fft_len/2;

                pts = av_rescale_q(pts, av_make_q(1, inlink->sample_rate), outlink->time_base);

                if (FFABS(pts - out->pts) > PTS_TOLERANCE) {

                    av_log(ctx, AV_LOG_DEBUG, "changing pts from %"PRId64" (%.3f) to %"PRId64" (%.3f).\n",

                           out->pts, out->pts * av_q2d(outlink->time_base),

                           pts, pts * av_q2d(outlink->time_base));

                    out->pts = pts;

                    s->next_pts = pts + PTS_STEP;

                }

                ret = ff_filter_frame(outlink, out);

                if (ret < 0) {

                    av_frame_free(&insamples);

                    return ret;

                }

                out = NULL;

            }

            step = s->step + (s->step_frac.num + s->remaining_frac) / s->step_frac.den;

            s->remaining_frac = (s->step_frac.num + s->remaining_frac) % s->step_frac.den;

            for (m = 0; m < s->fft_len-step; m++)

                s->fft_data[m] = s->fft_data[m+step];

            s->remaining_fill = step;

        } else {

            for (m = 0; m < remaining; m++) {

                s->fft_data[j+m].re = audio_data[2*(i+m)];

                s->fft_data[j+m].im = audio_data[2*(i+m)+1];

            }

            s->remaining_fill -= remaining;

            remaining = 0;

        }

    }

    av_frame_free(&insamples);

    return 0;

}
