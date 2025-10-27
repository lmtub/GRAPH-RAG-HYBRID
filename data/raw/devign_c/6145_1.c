static int compand_delay(AVFilterContext *ctx, AVFrame *frame)

{

    CompandContext *s = ctx->priv;

    AVFilterLink *inlink = ctx->inputs[0];

    const int channels = inlink->channels;

    const int nb_samples = frame->nb_samples;

    int chan, i, av_uninit(dindex), oindex, av_uninit(count);

    AVFrame *out_frame = NULL;



    av_assert1(channels > 0); /* would corrupt delay_count and delay_index */



    for (chan = 0; chan < channels; chan++) {

        const double *src = (double *)frame->extended_data[chan];

        double *dbuf = (double *)s->delayptrs[chan];

        ChanParam *cp = &s->channels[chan];

        double *dst;



        count  = s->delay_count;

        dindex = s->delay_index;

        for (i = 0, oindex = 0; i < nb_samples; i++) {

            const double in = src[i];

            update_volume(cp, fabs(in));



            if (count >= s->delay_samples) {

                if (!out_frame) {

                    out_frame = ff_get_audio_buffer(inlink, nb_samples - i);

                    if (!out_frame)

                        return AVERROR(ENOMEM);

                    av_frame_copy_props(out_frame, frame);

                    out_frame->pts = s->pts;

                    s->pts += av_rescale_q(nb_samples - i, (AVRational){1, inlink->sample_rate}, inlink->time_base);

                }



                dst = (double *)out_frame->extended_data[chan];

                dst[oindex++] = av_clipd(dbuf[dindex] * get_volume(s, cp->volume), -1, 1);

            } else {

                count++;

            }



            dbuf[dindex] = in;

            dindex = MOD(dindex + 1, s->delay_samples);

        }

    }



    s->delay_count = count;

    s->delay_index = dindex;



    av_frame_free(&frame);

    return out_frame ? ff_filter_frame(ctx->outputs[0], out_frame) : 0;

}
