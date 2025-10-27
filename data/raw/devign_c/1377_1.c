static int opus_decode_frame(OpusStreamContext *s, const uint8_t *data, int size)

{

    int samples    = s->packet.frame_duration;

    int redundancy = 0;

    int redundancy_size, redundancy_pos;

    int ret, i, consumed;

    int delayed_samples = s->delayed_samples;



    ret = opus_rc_init(&s->rc, data, size);

    if (ret < 0)

        return ret;



    /* decode the silk frame */

    if (s->packet.mode == OPUS_MODE_SILK || s->packet.mode == OPUS_MODE_HYBRID) {

        if (!swr_is_initialized(s->swr)) {

            ret = opus_init_resample(s);

            if (ret < 0)

                return ret;

        }



        samples = ff_silk_decode_superframe(s->silk, &s->rc, s->silk_output,

                                            FFMIN(s->packet.bandwidth, OPUS_BANDWIDTH_WIDEBAND),

                                            s->packet.stereo + 1,

                                            silk_frame_duration_ms[s->packet.config]);

        if (samples < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "Error decoding a SILK frame.\n");

            return samples;

        }

        samples = swr_convert(s->swr,

                              (uint8_t**)s->out, s->packet.frame_duration,

                              (const uint8_t**)s->silk_output, samples);

        if (samples < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "Error resampling SILK data.\n");

            return samples;

        }


        s->delayed_samples += s->packet.frame_duration - samples;

    } else

        ff_silk_flush(s->silk);



    // decode redundancy information

    consumed = opus_rc_tell(&s->rc);

    if (s->packet.mode == OPUS_MODE_HYBRID && consumed + 37 <= size * 8)

        redundancy = opus_rc_p2model(&s->rc, 12);

    else if (s->packet.mode == OPUS_MODE_SILK && consumed + 17 <= size * 8)

        redundancy = 1;



    if (redundancy) {

        redundancy_pos = opus_rc_p2model(&s->rc, 1);



        if (s->packet.mode == OPUS_MODE_HYBRID)

            redundancy_size = opus_rc_unimodel(&s->rc, 256) + 2;

        else

            redundancy_size = size - (consumed + 7) / 8;

        size -= redundancy_size;

        if (size < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "Invalid redundancy frame size.\n");

            return AVERROR_INVALIDDATA;

        }



        if (redundancy_pos) {

            ret = opus_decode_redundancy(s, data + size, redundancy_size);

            if (ret < 0)

                return ret;

            ff_celt_flush(s->celt);

        }

    }



    /* decode the CELT frame */

    if (s->packet.mode == OPUS_MODE_CELT || s->packet.mode == OPUS_MODE_HYBRID) {

        float *out_tmp[2] = { s->out[0], s->out[1] };

        float **dst = (s->packet.mode == OPUS_MODE_CELT) ?

                      out_tmp : s->celt_output;

        int celt_output_samples = samples;

        int delay_samples = av_audio_fifo_size(s->celt_delay);



        if (delay_samples) {

            if (s->packet.mode == OPUS_MODE_HYBRID) {

                av_audio_fifo_read(s->celt_delay, (void**)s->celt_output, delay_samples);



                for (i = 0; i < s->output_channels; i++) {

                    s->fdsp->vector_fmac_scalar(out_tmp[i], s->celt_output[i], 1.0,

                                                delay_samples);

                    out_tmp[i] += delay_samples;

                }

                celt_output_samples -= delay_samples;

            } else {

                av_log(s->avctx, AV_LOG_WARNING,

                       "Spurious CELT delay samples present.\n");

                av_audio_fifo_drain(s->celt_delay, delay_samples);

                if (s->avctx->err_recognition & AV_EF_EXPLODE)

                    return AVERROR_BUG;

            }

        }



        opus_raw_init(&s->rc, data + size, size);



        ret = ff_celt_decode_frame(s->celt, &s->rc, dst,

                                   s->packet.stereo + 1,

                                   s->packet.frame_duration,

                                   (s->packet.mode == OPUS_MODE_HYBRID) ? 17 : 0,

                                   celt_band_end[s->packet.bandwidth]);

        if (ret < 0)

            return ret;



        if (s->packet.mode == OPUS_MODE_HYBRID) {

            int celt_delay = s->packet.frame_duration - celt_output_samples;

            void *delaybuf[2] = { s->celt_output[0] + celt_output_samples,

                                  s->celt_output[1] + celt_output_samples };



            for (i = 0; i < s->output_channels; i++) {

                s->fdsp->vector_fmac_scalar(out_tmp[i],

                                            s->celt_output[i], 1.0,

                                            celt_output_samples);

            }



            ret = av_audio_fifo_write(s->celt_delay, delaybuf, celt_delay);

            if (ret < 0)

                return ret;

        }

    } else

        ff_celt_flush(s->celt);



    if (s->redundancy_idx) {

        for (i = 0; i < s->output_channels; i++)

            opus_fade(s->out[i], s->out[i],

                      s->redundancy_output[i] + 120 + s->redundancy_idx,

                      ff_celt_window2 + s->redundancy_idx, 120 - s->redundancy_idx);

        s->redundancy_idx = 0;

    }

    if (redundancy) {

        if (!redundancy_pos) {

            ff_celt_flush(s->celt);

            ret = opus_decode_redundancy(s, data + size, redundancy_size);

            if (ret < 0)

                return ret;



            for (i = 0; i < s->output_channels; i++) {

                opus_fade(s->out[i] + samples - 120 + delayed_samples,

                          s->out[i] + samples - 120 + delayed_samples,

                          s->redundancy_output[i] + 120,

                          ff_celt_window2, 120 - delayed_samples);

                if (delayed_samples)

                    s->redundancy_idx = 120 - delayed_samples;

            }

        } else {

            for (i = 0; i < s->output_channels; i++) {

                memcpy(s->out[i] + delayed_samples, s->redundancy_output[i], 120 * sizeof(float));

                opus_fade(s->out[i] + 120 + delayed_samples,

                          s->redundancy_output[i] + 120,

                          s->out[i] + 120 + delayed_samples,

                          ff_celt_window2, 120);

            }

        }

    }



    return samples;

}