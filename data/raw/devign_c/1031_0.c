static int theora_decode_header(AVCodecContext *avctx, GetBitContext gb)

{

    Vp3DecodeContext *s = avctx->priv_data;



    s->theora = get_bits_long(&gb, 24);

    av_log(avctx, AV_LOG_INFO, "Theora bitstream version %X\n", s->theora);



    /* 3.2.0 aka alpha3 has the same frame orientation as original vp3 */

    /* but previous versions have the image flipped relative to vp3 */

    if (s->theora < 0x030200)

    {

        s->flipped_image = 1;

        av_log(avctx, AV_LOG_DEBUG, "Old (<alpha3) Theora bitstream, flipped image\n");

    }



    s->width = get_bits(&gb, 16) << 4;

    s->height = get_bits(&gb, 16) << 4;



    if(avcodec_check_dimensions(avctx, s->width, s->height)){

        av_log(avctx, AV_LOG_ERROR, "Invalid dimensions (%dx%d)\n", s->width, s->height);

        s->width= s->height= 0;

        return -1;

    }



    if (s->theora >= 0x030400)

    {

        skip_bits(&gb, 32); /* total number of superblocks in a frame */

        // fixme, the next field is 36bits long

        skip_bits(&gb, 32); /* total number of blocks in a frame */

        skip_bits(&gb, 4); /* total number of blocks in a frame */

        skip_bits(&gb, 32); /* total number of macroblocks in a frame */



        skip_bits(&gb, 24); /* frame width */

        skip_bits(&gb, 24); /* frame height */

    }

    else

    {

        skip_bits(&gb, 24); /* frame width */

        skip_bits(&gb, 24); /* frame height */

    }



    skip_bits(&gb, 8); /* offset x */

    skip_bits(&gb, 8); /* offset y */



    skip_bits(&gb, 32); /* fps numerator */

    skip_bits(&gb, 32); /* fps denumerator */

    skip_bits(&gb, 24); /* aspect numerator */

    skip_bits(&gb, 24); /* aspect denumerator */



    if (s->theora < 0x030200)

        skip_bits(&gb, 5); /* keyframe frequency force */

    skip_bits(&gb, 8); /* colorspace */

    if (s->theora >= 0x030400)

        skip_bits(&gb, 2); /* pixel format: 420,res,422,444 */

    skip_bits(&gb, 24); /* bitrate */



    skip_bits(&gb, 6); /* quality hint */



    if (s->theora >= 0x030200)

    {

        skip_bits(&gb, 5); /* keyframe frequency force */



        if (s->theora < 0x030400)

            skip_bits(&gb, 5); /* spare bits */

    }



//    align_get_bits(&gb);



    avctx->width = s->width;

    avctx->height = s->height;



    return 0;

}
