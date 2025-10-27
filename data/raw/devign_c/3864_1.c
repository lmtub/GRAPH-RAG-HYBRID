static int theora_decode_header(AVCodecContext *avctx, GetBitContext *gb)

{

    Vp3DecodeContext *s = avctx->priv_data;

    int visible_width, visible_height, colorspace;

    int offset_x = 0, offset_y = 0;

    AVRational fps, aspect;



    s->theora = get_bits_long(gb, 24);

    av_log(avctx, AV_LOG_DEBUG, "Theora bitstream version %X\n", s->theora);



    /* 3.2.0 aka alpha3 has the same frame orientation as original vp3 */

    /* but previous versions have the image flipped relative to vp3 */

    if (s->theora < 0x030200)

    {

        s->flipped_image = 1;

        av_log(avctx, AV_LOG_DEBUG, "Old (<alpha3) Theora bitstream, flipped image\n");

    }



    visible_width  = s->width  = get_bits(gb, 16) << 4;

    visible_height = s->height = get_bits(gb, 16) << 4;



    if(av_image_check_size(s->width, s->height, 0, avctx)){

        av_log(avctx, AV_LOG_ERROR, "Invalid dimensions (%dx%d)\n", s->width, s->height);

        s->width= s->height= 0;

        return -1;

    }



    if (s->theora >= 0x030200) {

        visible_width  = get_bits_long(gb, 24);

        visible_height = get_bits_long(gb, 24);



        offset_x = get_bits(gb, 8); /* offset x */

        offset_y = get_bits(gb, 8); /* offset y, from bottom */

    }



    fps.num = get_bits_long(gb, 32);

    fps.den = get_bits_long(gb, 32);

    if (fps.num && fps.den) {

        av_reduce(&avctx->time_base.num, &avctx->time_base.den,

                  fps.den, fps.num, 1<<30);

    }



    aspect.num = get_bits_long(gb, 24);

    aspect.den = get_bits_long(gb, 24);

    if (aspect.num && aspect.den) {

        av_reduce(&avctx->sample_aspect_ratio.num,

                  &avctx->sample_aspect_ratio.den,

                  aspect.num, aspect.den, 1<<30);

    }



    if (s->theora < 0x030200)

        skip_bits(gb, 5); /* keyframe frequency force */

    colorspace = get_bits(gb, 8);

    skip_bits(gb, 24); /* bitrate */



    skip_bits(gb, 6); /* quality hint */



    if (s->theora >= 0x030200)

    {

        skip_bits(gb, 5); /* keyframe frequency force */

        avctx->pix_fmt = theora_pix_fmts[get_bits(gb, 2)];

        if (avctx->pix_fmt == AV_PIX_FMT_NONE) {

            av_log(avctx, AV_LOG_ERROR, "Invalid pixel format\n");

            return AVERROR_INVALIDDATA;

        }

        skip_bits(gb, 3); /* reserved */

    }



//    align_get_bits(gb);



    if (   visible_width  <= s->width  && visible_width  > s->width-16

        && visible_height <= s->height && visible_height > s->height-16

        && !offset_x && (offset_y == s->height - visible_height))

        avcodec_set_dimensions(avctx, visible_width, visible_height);

    else

        avcodec_set_dimensions(avctx, s->width, s->height);



    if (colorspace == 1) {

        avctx->color_primaries = AVCOL_PRI_BT470M;

    } else if (colorspace == 2) {

        avctx->color_primaries = AVCOL_PRI_BT470BG;

    }

    if (colorspace == 1 || colorspace == 2) {

        avctx->colorspace = AVCOL_SPC_BT470BG;

        avctx->color_trc  = AVCOL_TRC_BT709;

    }



    return 0;

}
