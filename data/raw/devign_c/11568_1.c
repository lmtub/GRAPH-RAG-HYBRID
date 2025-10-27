static int mpeg_decode_postinit(AVCodecContext *avctx)

{

    Mpeg1Context *s1  = avctx->priv_data;

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    uint8_t old_permutation[64];

    int ret;



    if (avctx->codec_id == AV_CODEC_ID_MPEG1VIDEO) {

        // MPEG-1 aspect

        avctx->sample_aspect_ratio = av_d2q(1.0 / ff_mpeg1_aspect[s->aspect_ratio_info], 255);

    } else { // MPEG-2

        // MPEG-2 aspect

        if (s->aspect_ratio_info > 1) {

            AVRational dar =

                av_mul_q(av_div_q(ff_mpeg2_aspect[s->aspect_ratio_info],

                                  (AVRational) { s1->pan_scan.width,

                                                 s1->pan_scan.height }),

                         (AVRational) { s->width, s->height });



            /* We ignore the spec here and guess a bit as reality does not

             * match the spec, see for example res_change_ffmpeg_aspect.ts

             * and sequence-display-aspect.mpg.

             * issue1613, 621, 562 */

            if ((s1->pan_scan.width == 0) || (s1->pan_scan.height == 0) ||

                (av_cmp_q(dar, (AVRational) { 4, 3 }) &&

                 av_cmp_q(dar, (AVRational) { 16, 9 }))) {

                s->avctx->sample_aspect_ratio =

                    av_div_q(ff_mpeg2_aspect[s->aspect_ratio_info],

                             (AVRational) { s->width, s->height });

            } else {

                s->avctx->sample_aspect_ratio =

                    av_div_q(ff_mpeg2_aspect[s->aspect_ratio_info],

                             (AVRational) { s1->pan_scan.width, s1->pan_scan.height });

// issue1613 4/3 16/9 -> 16/9

// res_change_ffmpeg_aspect.ts 4/3 225/44 ->4/3

// widescreen-issue562.mpg 4/3 16/9 -> 16/9

//                s->avctx->sample_aspect_ratio = av_mul_q(s->avctx->sample_aspect_ratio, (AVRational) {s->width, s->height});

                ff_dlog(avctx, "aspect A %d/%d\n",

                        ff_mpeg2_aspect[s->aspect_ratio_info].num,

                        ff_mpeg2_aspect[s->aspect_ratio_info].den);

                ff_dlog(avctx, "aspect B %d/%d\n", s->avctx->sample_aspect_ratio.num,

                        s->avctx->sample_aspect_ratio.den);

            }

        } else {

            s->avctx->sample_aspect_ratio =

                ff_mpeg2_aspect[s->aspect_ratio_info];

        }

    } // MPEG-2



    if (av_image_check_sar(s->width, s->height,

                           avctx->sample_aspect_ratio) < 0) {

        av_log(avctx, AV_LOG_WARNING, "ignoring invalid SAR: %u/%u\n",

                avctx->sample_aspect_ratio.num,

                avctx->sample_aspect_ratio.den);

        avctx->sample_aspect_ratio = (AVRational){ 0, 1 };

    }



    if ((s1->mpeg_enc_ctx_allocated == 0)                   ||

        avctx->coded_width       != s->width                ||

        avctx->coded_height      != s->height               ||

        s1->save_width           != s->width                ||

        s1->save_height          != s->height               ||

        av_cmp_q(s1->save_aspect, s->avctx->sample_aspect_ratio) ||

        (s1->save_progressive_seq != s->progressive_sequence && FFALIGN(s->height, 16) != FFALIGN(s->height, 32)) ||

        0) {

        if (s1->mpeg_enc_ctx_allocated) {

            ParseContext pc = s->parse_context;

            s->parse_context.buffer = 0;

            ff_mpv_common_end(s);

            s->parse_context = pc;

            s1->mpeg_enc_ctx_allocated = 0;

        }



        ret = ff_set_dimensions(avctx, s->width, s->height);

        if (ret < 0)

            return ret;



        if (avctx->codec_id == AV_CODEC_ID_MPEG2VIDEO && s->bit_rate) {

            avctx->rc_max_rate = s->bit_rate;

        } else if (avctx->codec_id == AV_CODEC_ID_MPEG1VIDEO && s->bit_rate &&

                   (s->bit_rate != 0x3FFFF*400 || s->vbv_delay != 0xFFFF)) {

            avctx->bit_rate = s->bit_rate;

        }

        s1->save_aspect          = s->avctx->sample_aspect_ratio;

        s1->save_width           = s->width;

        s1->save_height          = s->height;

        s1->save_progressive_seq = s->progressive_sequence;



        /* low_delay may be forced, in this case we will have B-frames

         * that behave like P-frames. */

        avctx->has_b_frames = !s->low_delay;



        if (avctx->codec_id == AV_CODEC_ID_MPEG1VIDEO) {

            // MPEG-1 fps

            avctx->framerate = ff_mpeg12_frame_rate_tab[s->frame_rate_index];

            avctx->ticks_per_frame     = 1;



            avctx->chroma_sample_location = AVCHROMA_LOC_CENTER;

        } else { // MPEG-2

            // MPEG-2 fps

            av_reduce(&s->avctx->framerate.num,

                      &s->avctx->framerate.den,

                      ff_mpeg12_frame_rate_tab[s->frame_rate_index].num * s1->frame_rate_ext.num,

                      ff_mpeg12_frame_rate_tab[s->frame_rate_index].den * s1->frame_rate_ext.den,

                      1 << 30);

            avctx->ticks_per_frame = 2;



            switch (s->chroma_format) {

            case 1: avctx->chroma_sample_location = AVCHROMA_LOC_LEFT; break;

            case 2:

            case 3: avctx->chroma_sample_location = AVCHROMA_LOC_TOPLEFT; break;

            default: av_assert0(0);

            }

        } // MPEG-2



        avctx->pix_fmt = mpeg_get_pixelformat(avctx);

        setup_hwaccel_for_pixfmt(avctx);



        /* Quantization matrices may need reordering

         * if DCT permutation is changed. */

        memcpy(old_permutation, s->idsp.idct_permutation, 64 * sizeof(uint8_t));



        ff_mpv_idct_init(s);

        if ((ret = ff_mpv_common_init(s)) < 0)

            return ret;



        quant_matrix_rebuild(s->intra_matrix,        old_permutation, s->idsp.idct_permutation);

        quant_matrix_rebuild(s->inter_matrix,        old_permutation, s->idsp.idct_permutation);

        quant_matrix_rebuild(s->chroma_intra_matrix, old_permutation, s->idsp.idct_permutation);

        quant_matrix_rebuild(s->chroma_inter_matrix, old_permutation, s->idsp.idct_permutation);



        s1->mpeg_enc_ctx_allocated = 1;

    }

    return 0;

}
