static int vc9_decode_init(AVCodecContext *avctx)

{

    VC9Context *v = avctx->priv_data;

    MpegEncContext *s = &v->s;

    GetBitContext gb;



    if (!avctx->extradata_size || !avctx->extradata) return -1;

    avctx->pix_fmt = PIX_FMT_YUV420P;

    v->s.avctx = avctx;



    if(ff_h263_decode_init(avctx) < 0)

        return -1;

    if (vc9_init_common(v) < 0) return -1;



    avctx->coded_width = avctx->width;

    avctx->coded_height = avctx->height;

    if (avctx->codec_id == CODEC_ID_WMV3)

    {

        int count = 0;



        // looks like WMV3 has a sequence header stored in the extradata

        // advanced sequence header may be before the first frame

        // the last byte of the extradata is a version number, 1 for the

        // samples we can decode



        init_get_bits(&gb, avctx->extradata, avctx->extradata_size);

        

        decode_sequence_header(avctx, &gb);



        count = avctx->extradata_size*8 - get_bits_count(&gb);

        if (count>0)

        {

            av_log(avctx, AV_LOG_INFO, "Extra data: %i bits left, value: %X\n",

                   count, get_bits(&gb, count));

        }

        else

        {

            av_log(avctx, AV_LOG_INFO, "Read %i bits in overflow\n", -count);

        }

    }

    avctx->has_b_frames= !!(avctx->max_b_frames);



    s->mb_width = (avctx->coded_width+15)>>4;

    s->mb_height = (avctx->coded_height+15)>>4;



    /* Allocate mb bitplanes */

    if (alloc_bitplane(&v->mv_type_mb_plane, s->mb_width, s->mb_height) < 0)

        return -1;

    if (alloc_bitplane(&v->mv_type_mb_plane, s->mb_width, s->mb_height) < 0)

        return -1;

    if (alloc_bitplane(&v->skip_mb_plane, s->mb_width, s->mb_height) < 0)

        return -1;

    if (alloc_bitplane(&v->direct_mb_plane, s->mb_width, s->mb_height) < 0)

        return -1;



    /* For predictors */

    v->previous_line_cbpcy = (uint8_t *)av_malloc(s->mb_stride*4);

    if (!v->previous_line_cbpcy) return -1;



#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

    {

        if (alloc_bitplane(&v->over_flags_plane, s->mb_width, s->mb_height) < 0)

            return -1;

        if (alloc_bitplane(&v->ac_pred_plane, s->mb_width, s->mb_height) < 0)

            return -1;

    }

#endif



    return 0;

    }
