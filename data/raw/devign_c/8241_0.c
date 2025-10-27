static int dca_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame_ptr, AVPacket *avpkt)

{

    AVFrame *frame     = data;

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;



    int lfe_samples;

    int num_core_channels = 0;

    int i, ret;

    float  **samples_flt;

    DCAContext *s = avctx->priv_data;

    int channels, full_channels;

    int core_ss_end;





    s->xch_present = 0;



    s->dca_buffer_size = ff_dca_convert_bitstream(buf, buf_size, s->dca_buffer,

                                                  DCA_MAX_FRAME_SIZE + DCA_MAX_EXSS_HEADER_SIZE);

    if (s->dca_buffer_size == AVERROR_INVALIDDATA) {

        av_log(avctx, AV_LOG_ERROR, "Not a valid DCA frame\n");

        return AVERROR_INVALIDDATA;

    }



    init_get_bits(&s->gb, s->dca_buffer, s->dca_buffer_size * 8);

    if ((ret = dca_parse_frame_header(s)) < 0) {

        //seems like the frame is corrupt, try with the next one

        return ret;

    }

    //set AVCodec values with parsed data

    avctx->sample_rate = s->sample_rate;

    avctx->bit_rate    = s->bit_rate;



    s->profile = FF_PROFILE_DTS;



    for (i = 0; i < (s->sample_blocks / 8); i++) {

        if ((ret = dca_decode_block(s, 0, i))) {

            av_log(avctx, AV_LOG_ERROR, "error decoding block\n");

            return ret;

        }

    }



    /* record number of core channels incase less than max channels are requested */

    num_core_channels = s->prim_channels;



    if (s->ext_coding)

        s->core_ext_mask = dca_ext_audio_descr_mask[s->ext_descr];

    else

        s->core_ext_mask = 0;



    core_ss_end = FFMIN(s->frame_size, s->dca_buffer_size) * 8;



    /* only scan for extensions if ext_descr was unknown or indicated a

     * supported XCh extension */

    if (s->core_ext_mask < 0 || s->core_ext_mask & DCA_EXT_XCH) {



        /* if ext_descr was unknown, clear s->core_ext_mask so that the

         * extensions scan can fill it up */

        s->core_ext_mask = FFMAX(s->core_ext_mask, 0);



        /* extensions start at 32-bit boundaries into bitstream */

        skip_bits_long(&s->gb, (-get_bits_count(&s->gb)) & 31);



        while (core_ss_end - get_bits_count(&s->gb) >= 32) {

            uint32_t bits = get_bits_long(&s->gb, 32);



            switch (bits) {

            case 0x5a5a5a5a: {

                int ext_amode, xch_fsize;



                s->xch_base_channel = s->prim_channels;



                /* validate sync word using XCHFSIZE field */

                xch_fsize = show_bits(&s->gb, 10);

                if ((s->frame_size != (get_bits_count(&s->gb) >> 3) - 4 + xch_fsize) &&

                    (s->frame_size != (get_bits_count(&s->gb) >> 3) - 4 + xch_fsize + 1))

                    continue;



                /* skip length-to-end-of-frame field for the moment */

                skip_bits(&s->gb, 10);



                s->core_ext_mask |= DCA_EXT_XCH;



                /* extension amode(number of channels in extension) should be 1 */

                /* AFAIK XCh is not used for more channels */

                if ((ext_amode = get_bits(&s->gb, 4)) != 1) {

                    av_log(avctx, AV_LOG_ERROR, "XCh extension amode %d not"

                           " supported!\n", ext_amode);

                    continue;

                }



                /* much like core primary audio coding header */

                dca_parse_audio_coding_header(s, s->xch_base_channel);



                for (i = 0; i < (s->sample_blocks / 8); i++)

                    if ((ret = dca_decode_block(s, s->xch_base_channel, i))) {

                        av_log(avctx, AV_LOG_ERROR, "error decoding XCh extension\n");

                        continue;

                    }



                s->xch_present = 1;

                break;

            }

            case 0x47004a03:

                /* XXCh: extended channels */

                /* usually found either in core or HD part in DTS-HD HRA streams,

                 * but not in DTS-ES which contains XCh extensions instead */

                s->core_ext_mask |= DCA_EXT_XXCH;

                break;



            case 0x1d95f262: {

                int fsize96 = show_bits(&s->gb, 12) + 1;

                if (s->frame_size != (get_bits_count(&s->gb) >> 3) - 4 + fsize96)

                    continue;



                av_log(avctx, AV_LOG_DEBUG, "X96 extension found at %d bits\n",

                       get_bits_count(&s->gb));

                skip_bits(&s->gb, 12);

                av_log(avctx, AV_LOG_DEBUG, "FSIZE96 = %d bytes\n", fsize96);

                av_log(avctx, AV_LOG_DEBUG, "REVNO = %d\n", get_bits(&s->gb, 4));



                s->core_ext_mask |= DCA_EXT_X96;

                break;

            }

            }



            skip_bits_long(&s->gb, (-get_bits_count(&s->gb)) & 31);

        }

    } else {

        /* no supported extensions, skip the rest of the core substream */

        skip_bits_long(&s->gb, core_ss_end - get_bits_count(&s->gb));

    }



    if (s->core_ext_mask & DCA_EXT_X96)

        s->profile = FF_PROFILE_DTS_96_24;

    else if (s->core_ext_mask & (DCA_EXT_XCH | DCA_EXT_XXCH))

        s->profile = FF_PROFILE_DTS_ES;



    /* check for ExSS (HD part) */

    if (s->dca_buffer_size - s->frame_size > 32 &&

        get_bits_long(&s->gb, 32) == DCA_HD_MARKER)

        dca_exss_parse_header(s);



    avctx->profile = s->profile;



    full_channels = channels = s->prim_channels + !!s->lfe;



    if (s->amode < 16) {

        avctx->channel_layout = dca_core_channel_layout[s->amode];



        if (s->prim_channels + !!s->lfe > 2 &&

            avctx->request_channel_layout == AV_CH_LAYOUT_STEREO) {

            /*

             * Neither the core's auxiliary data nor our default tables contain

             * downmix coefficients for the additional channel coded in the XCh

             * extension, so when we're doing a Stereo downmix, don't decode it.

             */

            s->xch_disable = 1;

        }



#if FF_API_REQUEST_CHANNELS

FF_DISABLE_DEPRECATION_WARNINGS

        if (s->xch_present && !s->xch_disable &&

            (!avctx->request_channels ||

             avctx->request_channels > num_core_channels + !!s->lfe)) {

FF_ENABLE_DEPRECATION_WARNINGS

#else

        if (s->xch_present && !s->xch_disable) {

#endif

            avctx->channel_layout |= AV_CH_BACK_CENTER;

            if (s->lfe) {

                avctx->channel_layout |= AV_CH_LOW_FREQUENCY;

                s->channel_order_tab = dca_channel_reorder_lfe_xch[s->amode];

            } else {

                s->channel_order_tab = dca_channel_reorder_nolfe_xch[s->amode];

            }

        } else {

            channels = num_core_channels + !!s->lfe;

            s->xch_present = 0; /* disable further xch processing */

            if (s->lfe) {

                avctx->channel_layout |= AV_CH_LOW_FREQUENCY;

                s->channel_order_tab = dca_channel_reorder_lfe[s->amode];

            } else

                s->channel_order_tab = dca_channel_reorder_nolfe[s->amode];

        }



        if (channels > !!s->lfe &&

            s->channel_order_tab[channels - 1 - !!s->lfe] < 0)

            return AVERROR_INVALIDDATA;



        if (s->prim_channels + !!s->lfe > 2 &&

            avctx->request_channel_layout == AV_CH_LAYOUT_STEREO) {

            channels = 2;

            s->output = s->prim_channels == 2 ? s->amode : DCA_STEREO;

            avctx->channel_layout = AV_CH_LAYOUT_STEREO;



            /* Stereo downmix coefficients

             *

             * The decoder can only downmix to 2-channel, so we need to ensure

             * embedded downmix coefficients are actually targeting 2-channel.

             */

            if (s->core_downmix && (s->core_downmix_amode == DCA_STEREO ||

                                    s->core_downmix_amode == DCA_STEREO_TOTAL)) {

                int sign, code;

                for (i = 0; i < s->prim_channels + !!s->lfe; i++) {

                    sign = s->core_downmix_codes[i][0] & 0x100 ? 1 : -1;

                    code = s->core_downmix_codes[i][0] & 0x0FF;

                    s->downmix_coef[i][0] = (!code ? 0.0f :

                                             sign * dca_dmixtable[code - 1]);

                    sign = s->core_downmix_codes[i][1] & 0x100 ? 1 : -1;

                    code = s->core_downmix_codes[i][1] & 0x0FF;

                    s->downmix_coef[i][1] = (!code ? 0.0f :

                                             sign * dca_dmixtable[code - 1]);

                }

                s->output = s->core_downmix_amode;

            } else {

                int am = s->amode & DCA_CHANNEL_MASK;

                if (am >= FF_ARRAY_ELEMS(dca_default_coeffs)) {

                    av_log(s->avctx, AV_LOG_ERROR,

                           "Invalid channel mode %d\n", am);

                    return AVERROR_INVALIDDATA;

                }

                if (s->prim_channels + !!s->lfe >

                    FF_ARRAY_ELEMS(dca_default_coeffs[0])) {

                    avpriv_request_sample(s->avctx, "Downmixing %d channels",

                                          s->prim_channels + !!s->lfe);

                    return AVERROR_PATCHWELCOME;

                }

                for (i = 0; i < s->prim_channels + !!s->lfe; i++) {

                    s->downmix_coef[i][0] = dca_default_coeffs[am][i][0];

                    s->downmix_coef[i][1] = dca_default_coeffs[am][i][1];

                }

            }

            av_dlog(s->avctx, "Stereo downmix coeffs:\n");

            for (i = 0; i < s->prim_channels + !!s->lfe; i++) {

                av_dlog(s->avctx, "L, input channel %d = %f\n", i,

                        s->downmix_coef[i][0]);

                av_dlog(s->avctx, "R, input channel %d = %f\n", i,

                        s->downmix_coef[i][1]);

            }

            av_dlog(s->avctx, "\n");

        }

    } else {

        av_log(avctx, AV_LOG_ERROR, "Non standard configuration %d !\n", s->amode);

        return AVERROR_INVALIDDATA;

    }

    avctx->channels = channels;



    /* get output buffer */

    frame->nb_samples = 256 * (s->sample_blocks / 8);

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }

    samples_flt = (float **)frame->extended_data;



    /* allocate buffer for extra channels if downmixing */

    if (avctx->channels < full_channels) {

        ret = av_samples_get_buffer_size(NULL, full_channels - channels,

                                         frame->nb_samples,

                                         avctx->sample_fmt, 0);

        if (ret < 0)

            return ret;



        av_fast_malloc(&s->extra_channels_buffer,

                       &s->extra_channels_buffer_size, ret);

        if (!s->extra_channels_buffer)

            return AVERROR(ENOMEM);



        ret = av_samples_fill_arrays((uint8_t **)s->extra_channels, NULL,

                                     s->extra_channels_buffer,

                                     full_channels - channels,

                                     frame->nb_samples, avctx->sample_fmt, 0);

        if (ret < 0)

            return ret;

    }



    /* filter to get final output */

    for (i = 0; i < (s->sample_blocks / 8); i++) {

        int ch;



        for (ch = 0; ch < channels; ch++)

            s->samples_chanptr[ch] = samples_flt[ch] + i * 256;

        for (; ch < full_channels; ch++)

            s->samples_chanptr[ch] = s->extra_channels[ch - channels] + i * 256;



        dca_filter_channels(s, i);



        /* If this was marked as a DTS-ES stream we need to subtract back- */

        /* channel from SL & SR to remove matrixed back-channel signal */

        if ((s->source_pcm_res & 1) && s->xch_present) {

            float *back_chan = s->samples_chanptr[s->channel_order_tab[s->xch_base_channel]];

            float *lt_chan   = s->samples_chanptr[s->channel_order_tab[s->xch_base_channel - 2]];

            float *rt_chan   = s->samples_chanptr[s->channel_order_tab[s->xch_base_channel - 1]];

            s->fdsp.vector_fmac_scalar(lt_chan, back_chan, -M_SQRT1_2, 256);

            s->fdsp.vector_fmac_scalar(rt_chan, back_chan, -M_SQRT1_2, 256);

        }

    }



    /* update lfe history */

    lfe_samples = 2 * s->lfe * (s->sample_blocks / 8);

    for (i = 0; i < 2 * s->lfe * 4; i++)

        s->lfe_data[i] = s->lfe_data[i + lfe_samples];



    /* AVMatrixEncoding

     *

     * DCA_STEREO_TOTAL (Lt/Rt) is equivalent to Dolby Surround */

    ret = ff_side_data_update_matrix_encoding(frame,

                                              (s->output & ~DCA_LFE) == DCA_STEREO_TOTAL ?

                                              AV_MATRIX_ENCODING_DOLBY : AV_MATRIX_ENCODING_NONE);

    if (ret < 0)

        return ret;



    *got_frame_ptr = 1;



    return buf_size;

}
