static int dca_decode_frame(AVCodecContext * avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;



    int lfe_samples;

    int num_core_channels = 0;

    int i;

    int xch_present = 0;

    int16_t *samples = data;

    DCAContext *s = avctx->priv_data;

    int channels;





    s->dca_buffer_size = dca_convert_bitstream(buf, buf_size, s->dca_buffer, DCA_MAX_FRAME_SIZE);

    if (s->dca_buffer_size == -1) {

        av_log(avctx, AV_LOG_ERROR, "Not a valid DCA frame\n");

        return -1;

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



    for (i = 0; i < (s->sample_blocks / 8); i++) {

        dca_decode_block(s, 0, i);

    }



    /* record number of core channels incase less than max channels are requested */

    num_core_channels = s->prim_channels;



    /* extensions start at 32-bit boundaries into bitstream */

    skip_bits_long(&s->gb, (-get_bits_count(&s->gb)) & 31);



    while(get_bits_left(&s->gb) >= 32) {

        uint32_t bits = get_bits_long(&s->gb, 32);



        switch(bits) {

        case 0x5a5a5a5a: {

            int ext_base_ch = s->prim_channels;

            int ext_amode;



            /* skip length-to-end-of-frame field for the moment */

            skip_bits(&s->gb, 10);



            /* extension amode should == 1, number of channels in extension */

            /* AFAIK XCh is not used for more channels */

            if ((ext_amode = get_bits(&s->gb, 4)) != 1) {

                av_log(avctx, AV_LOG_ERROR, "XCh extension amode %d not"

                       " supported!\n",ext_amode);

                continue;

            }



            /* much like core primary audio coding header */

            dca_parse_audio_coding_header(s, ext_base_ch);



            for (i = 0; i < (s->sample_blocks / 8); i++) {

                dca_decode_block(s, ext_base_ch, i);

            }



            xch_present = 1;

            break;

        }

        case 0x1d95f262:

            av_log(avctx, AV_LOG_DEBUG, "Possible X96 extension found at %d bits\n", get_bits_count(&s->gb));

            av_log(avctx, AV_LOG_DEBUG, "FSIZE96 = %d bytes\n", get_bits(&s->gb, 12)+1);

            av_log(avctx, AV_LOG_DEBUG, "REVNO = %d\n", get_bits(&s->gb, 4));

            break;

        }



        skip_bits_long(&s->gb, (-get_bits_count(&s->gb)) & 31);

    }



    channels = s->prim_channels + !!s->lfe;



    if (s->amode<16) {

        avctx->channel_layout = dca_core_channel_layout[s->amode];



        if (xch_present && (!avctx->request_channels ||

                            avctx->request_channels > num_core_channels)) {

            avctx->channel_layout |= CH_BACK_CENTER;

            if (s->lfe) {

                avctx->channel_layout |= CH_LOW_FREQUENCY;

                s->channel_order_tab = dca_channel_reorder_lfe_xch[s->amode];

            } else {

                s->channel_order_tab = dca_channel_reorder_nolfe_xch[s->amode];

            }

        } else {

            if (s->lfe) {

                avctx->channel_layout |= CH_LOW_FREQUENCY;

                s->channel_order_tab = dca_channel_reorder_lfe[s->amode];

            } else

                s->channel_order_tab = dca_channel_reorder_nolfe[s->amode];

        }



        if (s->prim_channels > 0 &&

            s->channel_order_tab[s->prim_channels - 1] < 0)

            return -1;



        if (avctx->request_channels == 2 && s->prim_channels > 2) {

            channels = 2;

            s->output = DCA_STEREO;

            avctx->channel_layout = CH_LAYOUT_STEREO;

        }

    } else {

        av_log(avctx, AV_LOG_ERROR, "Non standard configuration %d !\n",s->amode);

        return -1;

    }





    /* There is nothing that prevents a dts frame to change channel configuration

       but FFmpeg doesn't support that so only set the channels if it is previously

       unset. Ideally during the first probe for channels the crc should be checked

       and only set avctx->channels when the crc is ok. Right now the decoder could

       set the channels based on a broken first frame.*/

    if (!avctx->channels)

        avctx->channels = channels;



    if (*data_size < (s->sample_blocks / 8) * 256 * sizeof(int16_t) * channels)

        return -1;

    *data_size = 256 / 8 * s->sample_blocks * sizeof(int16_t) * channels;



    /* filter to get final output */

    for (i = 0; i < (s->sample_blocks / 8); i++) {

        dca_filter_channels(s, i);

        s->dsp.float_to_int16_interleave(samples, s->samples_chanptr, 256, channels);

        samples += 256 * channels;

    }



    /* update lfe history */

    lfe_samples = 2 * s->lfe * (s->sample_blocks / 8);

    for (i = 0; i < 2 * s->lfe * 4; i++) {

        s->lfe_data[i] = s->lfe_data[i + lfe_samples];

    }



    return buf_size;

}
