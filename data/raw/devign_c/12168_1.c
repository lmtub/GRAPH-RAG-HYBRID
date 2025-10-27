static int scan_for_extensions(AVCodecContext *avctx)

{

    DCAContext *s = avctx->priv_data;

    int core_ss_end, ret;



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

            int i;



            switch (bits) {

            case DCA_SYNCWORD_XCH: {

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

                    av_log(avctx, AV_LOG_ERROR,

                           "XCh extension amode %d not supported!\n",

                           ext_amode);

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

            case DCA_SYNCWORD_XXCH:

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

        get_bits_long(&s->gb, 32) == DCA_SYNCWORD_SUBSTREAM)

        ff_dca_exss_parse_header(s);



    return ret;

}
