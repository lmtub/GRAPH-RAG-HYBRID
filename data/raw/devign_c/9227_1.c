static void jpeg_put_comments(MpegEncContext *s)

{

    PutBitContext *p = &s->pb;

    int size;

    uint8_t *ptr;



    if (s->aspect_ratio_info /* && !lossless */)

    {

    /* JFIF header */

    put_marker(p, APP0);

    put_bits(p, 16, 16);

    put_string(p, "JFIF"); /* this puts the trailing zero-byte too */

    put_bits(p, 16, 0x0201); /* v 1.02 */

    put_bits(p, 8, 0); /* units type: 0 - aspect ratio */

    put_bits(p, 16, s->avctx->sample_aspect_ratio.num);

    put_bits(p, 16, s->avctx->sample_aspect_ratio.den);

    put_bits(p, 8, 0); /* thumbnail width */

    put_bits(p, 8, 0); /* thumbnail height */

    }



    /* comment */

    if(!(s->flags & CODEC_FLAG_BITEXACT)){

        put_marker(p, COM);

        flush_put_bits(p);

        ptr = pbBufPtr(p);

        put_bits(p, 16, 0); /* patched later */

        put_string(p, LIBAVCODEC_IDENT);

        size = strlen(LIBAVCODEC_IDENT)+3;

        ptr[0] = size >> 8;

        ptr[1] = size;

    }

}
