static int filter_frame(AVFilterLink *inlink, AVFrame *buf)

{

    AVFilterContext *ctx = inlink->dst;

    VolumeContext *vol    = inlink->dst->priv;

    AVFilterLink *outlink = inlink->dst->outputs[0];

    int nb_samples        = buf->nb_samples;

    AVFrame *out_buf;

    int64_t pos;

    AVFrameSideData *sd = av_frame_get_side_data(buf, AV_FRAME_DATA_REPLAYGAIN);

    int ret;



    if (sd && vol->replaygain != REPLAYGAIN_IGNORE) {

        if (vol->replaygain != REPLAYGAIN_DROP) {

            AVReplayGain *replaygain = (AVReplayGain*)sd->data;

            int32_t gain  = 100000;

            uint32_t peak = 100000;

            float g, p;



            if (vol->replaygain == REPLAYGAIN_TRACK &&

                replaygain->track_gain != INT32_MIN) {

                gain = replaygain->track_gain;



                if (replaygain->track_peak != 0)

                    peak = replaygain->track_peak;

            } else if (replaygain->album_gain != INT32_MIN) {

                gain = replaygain->album_gain;



                if (replaygain->album_peak != 0)

                    peak = replaygain->album_peak;

            } else {

                av_log(inlink->dst, AV_LOG_WARNING, "Both ReplayGain gain "

                       "values are unknown.\n");

            }

            g = gain / 100000.0f;

            p = peak / 100000.0f;



            av_log(inlink->dst, AV_LOG_VERBOSE,

                   "Using gain %f dB from replaygain side data.\n", g);



            vol->volume   = ff_exp10((g + vol->replaygain_preamp) / 20);

            if (vol->replaygain_noclip)

                vol->volume = FFMIN(vol->volume, 1.0 / p);

            vol->volume_i = (int)(vol->volume * 256 + 0.5);



            volume_init(vol);

        }

        av_frame_remove_side_data(buf, AV_FRAME_DATA_REPLAYGAIN);

    }



    if (isnan(vol->var_values[VAR_STARTPTS])) {

        vol->var_values[VAR_STARTPTS] = TS2D(buf->pts);

        vol->var_values[VAR_STARTT  ] = TS2T(buf->pts, inlink->time_base);

    }

    vol->var_values[VAR_PTS] = TS2D(buf->pts);

    vol->var_values[VAR_T  ] = TS2T(buf->pts, inlink->time_base);

    vol->var_values[VAR_N  ] = inlink->frame_count_out;



    pos = buf->pkt_pos;

    vol->var_values[VAR_POS] = pos == -1 ? NAN : pos;

    if (vol->eval_mode == EVAL_MODE_FRAME)

        set_volume(ctx);



    if (vol->volume == 1.0 || vol->volume_i == 256) {

        out_buf = buf;

        goto end;

    }



    /* do volume scaling in-place if input buffer is writable */

    if (av_frame_is_writable(buf)

            && (vol->precision != PRECISION_FIXED || vol->volume_i > 0)) {

        out_buf = buf;

    } else {

        out_buf = ff_get_audio_buffer(inlink, nb_samples);

        if (!out_buf)

            return AVERROR(ENOMEM);

        ret = av_frame_copy_props(out_buf, buf);

        if (ret < 0) {

            av_frame_free(&out_buf);

            av_frame_free(&buf);

            return ret;

        }

    }



    if (vol->precision != PRECISION_FIXED || vol->volume_i > 0) {

        int p, plane_samples;



        if (av_sample_fmt_is_planar(buf->format))

            plane_samples = FFALIGN(nb_samples, vol->samples_align);

        else

            plane_samples = FFALIGN(nb_samples * vol->channels, vol->samples_align);



        if (vol->precision == PRECISION_FIXED) {

            for (p = 0; p < vol->planes; p++) {

                vol->scale_samples(out_buf->extended_data[p],

                                   buf->extended_data[p], plane_samples,

                                   vol->volume_i);

            }

        } else if (av_get_packed_sample_fmt(vol->sample_fmt) == AV_SAMPLE_FMT_FLT) {

            for (p = 0; p < vol->planes; p++) {

                vol->fdsp->vector_fmul_scalar((float *)out_buf->extended_data[p],

                                             (const float *)buf->extended_data[p],

                                             vol->volume, plane_samples);

            }

        } else {

            for (p = 0; p < vol->planes; p++) {

                vol->fdsp->vector_dmul_scalar((double *)out_buf->extended_data[p],

                                             (const double *)buf->extended_data[p],

                                             vol->volume, plane_samples);

            }

        }

    }



    emms_c();



    if (buf != out_buf)

        av_frame_free(&buf);



end:

    vol->var_values[VAR_NB_CONSUMED_SAMPLES] += out_buf->nb_samples;

    return ff_filter_frame(outlink, out_buf);

}
