static int ac3_decode_frame(AVCodecContext * avctx, void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    AC3DecodeContext *s = avctx->priv_data;

    int16_t *out_samples = (int16_t *)data;

    int blk, ch, err;

    const uint8_t *channel_map;

    const float *output[AC3_MAX_CHANNELS];



    /* initialize the GetBitContext with the start of valid AC-3 Frame */

    if (s->input_buffer) {

        /* copy input buffer to decoder context to avoid reading past the end

           of the buffer, which can be caused by a damaged input stream. */

        memcpy(s->input_buffer, buf, FFMIN(buf_size, AC3_FRAME_BUFFER_SIZE));

        init_get_bits(&s->gbc, s->input_buffer, buf_size * 8);

    } else {

        init_get_bits(&s->gbc, buf, buf_size * 8);

    }



    /* parse the syncinfo */

    *data_size = 0;

    err = parse_frame_header(s);



    if (err) {

        switch(err) {

            case AAC_AC3_PARSE_ERROR_SYNC:

                av_log(avctx, AV_LOG_ERROR, "frame sync error\n");

                return -1;

            case AAC_AC3_PARSE_ERROR_BSID:

                av_log(avctx, AV_LOG_ERROR, "invalid bitstream id\n");

                break;

            case AAC_AC3_PARSE_ERROR_SAMPLE_RATE:

                av_log(avctx, AV_LOG_ERROR, "invalid sample rate\n");

                break;

            case AAC_AC3_PARSE_ERROR_FRAME_SIZE:

                av_log(avctx, AV_LOG_ERROR, "invalid frame size\n");

                break;

            case AAC_AC3_PARSE_ERROR_FRAME_TYPE:

                /* skip frame if CRC is ok. otherwise use error concealment. */

                /* TODO: add support for substreams and dependent frames */

                if(s->frame_type == EAC3_FRAME_TYPE_DEPENDENT || s->substreamid) {

                    av_log(avctx, AV_LOG_ERROR, "unsupported frame type : skipping frame\n");

                    return s->frame_size;

                } else {

                    av_log(avctx, AV_LOG_ERROR, "invalid frame type\n");

                }

                break;

            default:

                av_log(avctx, AV_LOG_ERROR, "invalid header\n");

                break;

        }

    } else {

        /* check that reported frame size fits in input buffer */

        if (s->frame_size > buf_size) {

            av_log(avctx, AV_LOG_ERROR, "incomplete frame\n");

            err = AAC_AC3_PARSE_ERROR_FRAME_SIZE;

        } else if (avctx->error_recognition >= FF_ER_CAREFUL) {

            /* check for crc mismatch */

            if (av_crc(av_crc_get_table(AV_CRC_16_ANSI), 0, &buf[2], s->frame_size-2)) {

                av_log(avctx, AV_LOG_ERROR, "frame CRC mismatch\n");

                err = AAC_AC3_PARSE_ERROR_CRC;

            }

        }

    }



    /* if frame is ok, set audio parameters */

    if (!err) {

        avctx->sample_rate = s->sample_rate;

        avctx->bit_rate = s->bit_rate;



        /* channel config */

        s->out_channels = s->channels;

        s->output_mode = s->channel_mode;

        if(s->lfe_on)

            s->output_mode |= AC3_OUTPUT_LFEON;

        if (avctx->request_channels > 0 && avctx->request_channels <= 2 &&

                avctx->request_channels < s->channels) {

            s->out_channels = avctx->request_channels;

            s->output_mode  = avctx->request_channels == 1 ? AC3_CHMODE_MONO : AC3_CHMODE_STEREO;

            s->channel_layout = ff_ac3_channel_layout_tab[s->output_mode];

        }

        avctx->channels = s->out_channels;

        avctx->channel_layout = s->channel_layout;



        /* set downmixing coefficients if needed */

        if(s->channels != s->out_channels && !((s->output_mode & AC3_OUTPUT_LFEON) &&

                s->fbw_channels == s->out_channels)) {

            set_downmix_coeffs(s);

        }

    } else if (!s->out_channels) {

        s->out_channels = avctx->channels;

        if(s->out_channels < s->channels)

            s->output_mode  = s->out_channels == 1 ? AC3_CHMODE_MONO : AC3_CHMODE_STEREO;

    }



    /* decode the audio blocks */

    channel_map = ff_ac3_dec_channel_map[s->output_mode & ~AC3_OUTPUT_LFEON][s->lfe_on];

    for (ch = 0; ch < s->out_channels; ch++)

        output[ch] = s->output[channel_map[ch]];

    for (blk = 0; blk < s->num_blocks; blk++) {

        if (!err && decode_audio_block(s, blk)) {

            av_log(avctx, AV_LOG_ERROR, "error decoding the audio block\n");

            err = 1;

        }

        s->fmt_conv.float_to_int16_interleave(out_samples, output, 256, s->out_channels);

        out_samples += 256 * s->out_channels;

    }

    *data_size = s->num_blocks * 256 * avctx->channels * sizeof (int16_t);

    return FFMIN(buf_size, s->frame_size);

}
