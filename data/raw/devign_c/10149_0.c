static int dca_decode_frame(AVCodecContext * avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;



    int lfe_samples;

    int num_core_channels = 0;

    int i;

    float   *samples_flt = data;

    int16_t *samples_s16 = data;

    int out_size;

    DCAContext *s = avctx->priv_data;

    int channels;

    int core_ss_end;





    s->xch_present = 0;



    s->dca_buffer_size = dca_convert_bitstream(buf, buf_size, s->dca_buffer,

                                               DCA_MAX_FRAME_SIZE + DCA_MAX_EXSS_HEADER_SIZE);

    if (s->dca_buffer_size == AVERROR_INVALIDDATA) {

        av_log(avctx, AV_LOG_ERROR, "Not a valid DCA frame\n");

        return AVERROR_INVALIDDATA;

    }



    init_get_bits(&s->gb, s->dca_buffer, s->dca_buffer_size * 8);

    if (dca_parse_frame_header(s) < 0) {

        //seems like the frame is corrupt, try with the next one

        *data_size=0;

        return buf_size;

    }

    //set AVCodec values with parsed data

    avctx->sample_rate = s->sample_rate;

    avctx->bit_rate = s->bit_rate;

    avctx->frame_size = s->sample_blocks * 32;



    s->profile = FF_PROFILE_DTS;



    for (i = 0; i < (s->sample_blocks / 8); i++) {

        dca_decode_block(s, 0, i);

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



    while(core_ss_end - get_bits_count(&s->gb) >= 32) {

        uint32_t bits = get_bits_long(&s->gb, 32);



        switch(bits) {

        case 0x5a5a5a5a: {

            int ext_amode, xch_fsize;



            s->xch_base_channel = s->prim_channels;



            /* validate sync word using XCHFSIZE field */

            xch_fsize = show_bits(&s->gb, 10);

            if((s->frame_size != (get_bits_count(&s->gb) >> 3) - 4 + xch_fsize) &&

               (s->frame_size != (get_bits_count(&s->gb) >> 3) - 4 + xch_fsize + 1))

                continue;



            /* skip length-to-end-of-frame field for the moment */

            skip_bits(&s->gb, 10);



            s->core_ext_mask |= DCA_EXT_XCH;



            /* extension amode should == 1, number of channels in extension */

            /* AFAIK XCh is not used for more channels */

            if ((ext_amode = get_bits(&s->gb, 4)) != 1) {

                av_log(avctx, AV_LOG_ERROR, "XCh extension amode %d not"

                       " supported!\n",ext_amode);

                continue;

            }



            /* much like core primary audio coding header */

            dca_parse_audio_coding_header(s, s->xch_base_channel);



            for (i = 0; i < (s->sample_blocks / 8); i++) {

                dca_decode_block(s, s->xch_base_channel, i);

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



            av_log(avctx, AV_LOG_DEBUG, "X96 extension found at %d bits\n", get_bits_count(&s->gb));

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

    if (s->dca_buffer_size - s->frame_size > 32

        && get_bits_long(&s->gb, 32) == DCA_HD_MARKER)

        dca_exss_parse_header(s);



    avctx->profile = s->profile;



    channels = s->prim_channels + !!s->lfe;



    if (s->amode<16) {

        avctx->channel_layout = dca_core_channel_layout[s->amode];



        if (s->xch_present && (!avctx->request_channels ||

                               avctx->request_channels > num_core_channels + !!s->lfe)) {

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



        if (avctx->request_channels == 2 && s->prim_channels > 2) {

            channels = 2;

            s->output = DCA_STEREO;

            avctx->channel_layout = AV_CH_LAYOUT_STEREO;

        }

    } else {

        av_log(avctx, AV_LOG_ERROR, "Non standard configuration %d !\n",s->amode);

        return AVERROR_INVALIDDATA;

    }





    /* There is nothing that prevents a dts frame to change channel configuration

       but Libav doesn't support that so only set the channels if it is previously

       unset. Ideally during the first probe for channels the crc should be checked

       and only set avctx->channels when the crc is ok. Right now the decoder could

       set the channels based on a broken first frame.*/

    if (s->is_channels_set == 0) {

        s->is_channels_set = 1;

        avctx->channels = channels;

    }

    if (avctx->channels != channels) {

        av_log(avctx, AV_LOG_ERROR, "DCA decoder does not support number of "

               "channels changing in stream. Skipping frame.\n");

        return AVERROR_PATCHWELCOME;

    }



    out_size = 256 / 8 * s->sample_blocks * channels *

               av_get_bytes_per_sample(avctx->sample_fmt);

    if (*data_size < out_size)

        return AVERROR(EINVAL);

    *data_size = out_size;



    /* filter to get final output */

    for (i = 0; i < (s->sample_blocks / 8); i++) {

        dca_filter_channels(s, i);



        /* If this was marked as a DTS-ES stream we need to subtract back- */

        /* channel from SL & SR to remove matrixed back-channel signal */

        if((s->source_pcm_res & 1) && s->xch_present) {

            float* back_chan = s->samples + s->channel_order_tab[s->xch_base_channel] * 256;

            float* lt_chan   = s->samples + s->channel_order_tab[s->xch_base_channel - 2] * 256;

            float* rt_chan   = s->samples + s->channel_order_tab[s->xch_base_channel - 1] * 256;

            s->dsp.vector_fmac_scalar(lt_chan, back_chan, -M_SQRT1_2, 256);

            s->dsp.vector_fmac_scalar(rt_chan, back_chan, -M_SQRT1_2, 256);

        }



        if (avctx->sample_fmt == AV_SAMPLE_FMT_FLT) {

            s->fmt_conv.float_interleave(samples_flt, s->samples_chanptr, 256,

                                         channels);

            samples_flt += 256 * channels;

        } else {

            s->fmt_conv.float_to_int16_interleave(samples_s16,

                                                  s->samples_chanptr, 256,

                                                  channels);

            samples_s16 += 256 * channels;

        }

    }



    /* update lfe history */

    lfe_samples = 2 * s->lfe * (s->sample_blocks / 8);

    for (i = 0; i < 2 * s->lfe * 4; i++) {

        s->lfe_data[i] = s->lfe_data[i + lfe_samples];

    }



    return buf_size;

}
