static int mpeg4_decode_header(AVCodecParserContext *s1, AVCodecContext *avctx,

                               const uint8_t *buf, int buf_size)

{

    struct Mp4vParseContext *pc = s1->priv_data;

    Mpeg4DecContext *dec_ctx = &pc->dec_ctx;

    MpegEncContext *s = &dec_ctx->m;

    GetBitContext gb1, *gb = &gb1;

    int ret;



    s->avctx               = avctx;

    s->current_picture_ptr = &s->current_picture;



    if (avctx->extradata_size && pc->first_picture) {

        init_get_bits(gb, avctx->extradata, avctx->extradata_size * 8);

        ret = ff_mpeg4_decode_picture_header(dec_ctx, gb);



    }



    init_get_bits(gb, buf, 8 * buf_size);

    ret = ff_mpeg4_decode_picture_header(dec_ctx, gb);

    if (s->width && (!avctx->width || !avctx->height ||

                     !avctx->coded_width || !avctx->coded_height)) {

        ret = ff_set_dimensions(avctx, s->width, s->height);


            return ret;

    }

    if((s1->flags & PARSER_FLAG_USE_CODEC_TS) && s->avctx->time_base.den>0 && ret>=0){

        av_assert1(s1->pts == AV_NOPTS_VALUE);

        av_assert1(s1->dts == AV_NOPTS_VALUE);



        s1->pts = av_rescale_q(s->time, (AVRational){1, s->avctx->time_base.den}, (AVRational){1, 1200000});

    }



    s1->pict_type     = s->pict_type;

    pc->first_picture = 0;

    return ret;

}