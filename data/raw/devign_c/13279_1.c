static int dca_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame_ptr, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    int channel_mask;

    int channel_layout;

    int lfe_samples;

    int num_core_channels = 0;

    int i, ret;

    float **samples_flt;

    float *src_chan;

    float *dst_chan;

    DCAContext *s = avctx->priv_data;

    int core_ss_end;

    int channels, full_channels;

    float scale;

    int achan;

    int chset;

    int mask;

    int lavc;

    int posn;

    int j, k;

    int endch;



    s->xch_present = 0;



    s->dca_buffer_size = ff_dca_convert_bitstream(buf, buf_size, s->dca_buffer,

                                                  DCA_MAX_FRAME_SIZE + DCA_MAX_EXSS_HEADER_SIZE);

    if (s->dca_buffer_size == AVERROR_INVALIDDATA) {

        av_log(avctx, AV_LOG_ERROR, "Not a valid DCA frame\n");


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

    if (s->core_ext_mask < 0 || s->core_ext_mask & (DCA_EXT_XCH | DCA_EXT_XXCH)) {



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



                if (s->xch_base_channel < 2) {

                    av_log_ask_for_sample(avctx, "XCh with fewer than 2 base channels is not supported\n");

                    continue;

                }



                /* much like core primary audio coding header */

                dca_parse_audio_coding_header(s, s->xch_base_channel, 0);



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

                dca_xxch_decode_frame(s);

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



    /* If we have XXCH then the channel layout is managed differently */

    /* note that XLL will also have another way to do things */

    if (!(s->core_ext_mask & DCA_EXT_XXCH)

        || (s->core_ext_mask & DCA_EXT_XXCH && avctx->request_channels > 0

            && avctx->request_channels

            < num_core_channels + !!s->lfe + s->xxch_chset_nch[0]))

    { /* xxx should also do MA extensions */

        if (s->amode < 16) {

            avctx->channel_layout = dca_core_channel_layout[s->amode];



            if (s->xch_present && (!avctx->request_channels ||

                                   avctx->request_channels

                                   > num_core_channels + !!s->lfe)) {

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




            if (av_get_channel_layout_nb_channels(avctx->channel_layout) != channels) {

                av_log(avctx, AV_LOG_ERROR, "Number of channels %d mismatches layout %d\n", channels, av_get_channel_layout_nb_channels(avctx->channel_layout));


            }



            if (avctx->request_channels == 2 && s->prim_channels > 2) {

                channels = 2;

                s->output = DCA_STEREO;

                avctx->channel_layout = AV_CH_LAYOUT_STEREO;

            }

            else if (avctx->request_channel_layout & AV_CH_LAYOUT_NATIVE) {

                static const int8_t dca_channel_order_native[9] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };

                s->channel_order_tab = dca_channel_order_native;

            }

            s->lfe_index = dca_lfe_index[s->amode];

        } else {

            av_log(avctx, AV_LOG_ERROR,

                   "Non standard configuration %d !\n", s->amode);


        }



        s->xxch_dmix_embedded = 0;

    } else {

        /* we only get here if an XXCH channel set can be added to the mix */

        channel_mask = s->xxch_core_spkmask;



        if (avctx->request_channels > 0

            && avctx->request_channels < s->prim_channels) {

            channels = num_core_channels + !!s->lfe;

            for (i = 0; i < s->xxch_chset && channels + s->xxch_chset_nch[i]

                                              <= avctx->request_channels; i++) {

                channels += s->xxch_chset_nch[i];

                channel_mask |= s->xxch_spk_masks[i];

            }

        } else {

            channels = s->prim_channels + !!s->lfe;

            for (i = 0; i < s->xxch_chset; i++) {

                channel_mask |= s->xxch_spk_masks[i];

            }

        }



        /* Given the DTS spec'ed channel mask, generate an avcodec version */

        channel_layout = 0;

        for (i = 0; i < s->xxch_nbits_spk_mask; ++i) {

            if (channel_mask & (1 << i)) {

                channel_layout |= map_xxch_to_native[i];

            }

        }



        /* make sure that we have managed to get equivelant dts/avcodec channel

         * masks in some sense -- unfortunately some channels could overlap */

        if (av_popcount(channel_mask) != av_popcount(channel_layout)) {

            av_log(avctx, AV_LOG_DEBUG,

                   "DTS-XXCH: Inconsistant avcodec/dts channel layouts\n");


        }



        avctx->channel_layout = channel_layout;



        if (!(avctx->request_channel_layout & AV_CH_LAYOUT_NATIVE)) {

            /* Estimate DTS --> avcodec ordering table */

            for (chset = -1, j = 0; chset < s->xxch_chset; ++chset) {

                mask = chset >= 0 ? s->xxch_spk_masks[chset]

                                  : s->xxch_core_spkmask;

                for (i = 0; i < s->xxch_nbits_spk_mask; i++) {

                    if (mask & ~(DCA_XXCH_LFE1 | DCA_XXCH_LFE2) & (1 << i)) {

                        lavc = map_xxch_to_native[i];

                        posn = av_popcount(channel_layout & (lavc - 1));

                        s->xxch_order_tab[j++] = posn;

                    }

                }

            }



            s->lfe_index = av_popcount(channel_layout & (AV_CH_LOW_FREQUENCY-1));

        } else { /* native ordering */

            for (i = 0; i < channels; i++)

                s->xxch_order_tab[i] = i;



            s->lfe_index = channels - 1;

        }



        s->channel_order_tab = s->xxch_order_tab;

    }



    if (avctx->channels != channels) {

        if (avctx->channels)

            av_log(avctx, AV_LOG_INFO, "Number of channels changed in DCA decoder (%d -> %d)\n", avctx->channels, channels);

        avctx->channels = channels;

    }



    /* get output buffer */

    s->frame.nb_samples = 256 * (s->sample_blocks / 8);

    if ((ret = ff_get_buffer(avctx, &s->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }

    samples_flt = (float  **) s->frame.extended_data;



    /* allocate buffer for extra channels if downmixing */

    if (avctx->channels < full_channels) {

        ret = av_samples_get_buffer_size(NULL, full_channels - channels,

                                         s->frame.nb_samples,

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

                                     s->frame.nb_samples, avctx->sample_fmt, 0);

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



        /* If stream contains XXCH, we might need to undo an embedded downmix */

        if (s->xxch_dmix_embedded) {

            /* Loop over channel sets in turn */

            ch = num_core_channels;

            for (chset = 0; chset < s->xxch_chset; chset++) {

                endch = ch + s->xxch_chset_nch[chset];

                mask = s->xxch_dmix_embedded;



                /* undo downmix */

                for (j = ch; j < endch; j++) {

                    if (mask & (1 << j)) { /* this channel has been mixed-out */

                        src_chan = s->samples_chanptr[s->channel_order_tab[j]];

                        for (k = 0; k < endch; k++) {

                            achan = s->channel_order_tab[k];

                            scale = s->xxch_dmix_coeff[j][k];

                            if (scale != 0.0) {

                                dst_chan = s->samples_chanptr[achan];

                                s->fdsp.vector_fmac_scalar(dst_chan, src_chan,

                                                           -scale, 256);

                            }

                        }

                    }

                }



                /* if a downmix has been embedded then undo the pre-scaling */

                if ((mask & (1 << ch)) && s->xxch_dmix_sf[chset] != 1.0f) {

                    scale = s->xxch_dmix_sf[chset];



                    for (j = 0; j < ch; j++) {

                        src_chan = s->samples_chanptr[s->channel_order_tab[j]];

                        for (k = 0; k < 256; k++)

                            src_chan[k] *= scale;

                    }



                    /* LFE channel is always part of core, scale if it exists */

                    if (s->lfe) {

                        src_chan = s->samples_chanptr[s->lfe_index];

                        for (k = 0; k < 256; k++)

                            src_chan[k] *= scale;

                    }

                }



                ch = endch;

            }



        }

    }



    /* update lfe history */

    lfe_samples = 2 * s->lfe * (s->sample_blocks / 8);

    for (i = 0; i < 2 * s->lfe * 4; i++)

        s->lfe_data[i] = s->lfe_data[i + lfe_samples];



    *got_frame_ptr    = 1;

    *(AVFrame *) data = s->frame;



    return buf_size;

}