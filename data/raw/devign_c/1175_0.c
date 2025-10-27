void ff_mjpeg_encode_picture_header(AVCodecContext *avctx, PutBitContext *pb,

                                    ScanTable *intra_scantable,

                                    uint16_t intra_matrix[64])

{

    int chroma_h_shift, chroma_v_shift;

    const int lossless = avctx->codec_id != AV_CODEC_ID_MJPEG;

    int hsample[3], vsample[3];



    av_pix_fmt_get_chroma_sub_sample(avctx->pix_fmt, &chroma_h_shift,

                                     &chroma_v_shift);



    if (avctx->codec->id == AV_CODEC_ID_LJPEG &&

        avctx->pix_fmt   == AV_PIX_FMT_BGR24) {

        vsample[0] = hsample[0] =

        vsample[1] = hsample[1] =

        vsample[2] = hsample[2] = 1;

    } else {

        vsample[0] = 2;

        vsample[1] = 2 >> chroma_v_shift;

        vsample[2] = 2 >> chroma_v_shift;

        hsample[0] = 2;

        hsample[1] = 2 >> chroma_h_shift;

        hsample[2] = 2 >> chroma_h_shift;

    }



    put_marker(pb, SOI);



    jpeg_put_comments(avctx, pb);



    jpeg_table_header(pb, intra_scantable, intra_matrix);



    switch (avctx->codec_id) {

    case AV_CODEC_ID_MJPEG:  put_marker(pb, SOF0 ); break;

    case AV_CODEC_ID_LJPEG:  put_marker(pb, SOF3 ); break;

    default: assert(0);

    }



    put_bits(pb, 16, 17);

    if (lossless && avctx->pix_fmt == AV_PIX_FMT_BGR24)

        put_bits(pb, 8, 9); /* 9 bits/component RCT */

    else

        put_bits(pb, 8, 8); /* 8 bits/component */

    put_bits(pb, 16, avctx->height);

    put_bits(pb, 16, avctx->width);

    put_bits(pb, 8, 3); /* 3 components */



    /* Y component */

    put_bits(pb, 8, 1); /* component number */

    put_bits(pb, 4, hsample[0]); /* H factor */

    put_bits(pb, 4, vsample[0]); /* V factor */

    put_bits(pb, 8, 0); /* select matrix */



    /* Cb component */

    put_bits(pb, 8, 2); /* component number */

    put_bits(pb, 4, hsample[1]); /* H factor */

    put_bits(pb, 4, vsample[1]); /* V factor */

    put_bits(pb, 8, 0); /* select matrix */



    /* Cr component */

    put_bits(pb, 8, 3); /* component number */

    put_bits(pb, 4, hsample[2]); /* H factor */

    put_bits(pb, 4, vsample[2]); /* V factor */

    put_bits(pb, 8, 0); /* select matrix */



    /* scan header */

    put_marker(pb, SOS);

    put_bits(pb, 16, 12); /* length */

    put_bits(pb, 8, 3); /* 3 components */



    /* Y component */

    put_bits(pb, 8, 1); /* index */

    put_bits(pb, 4, 0); /* DC huffman table index */

    put_bits(pb, 4, 0); /* AC huffman table index */



    /* Cb component */

    put_bits(pb, 8, 2); /* index */

    put_bits(pb, 4, 1); /* DC huffman table index */

    put_bits(pb, 4, lossless ? 0 : 1); /* AC huffman table index */



    /* Cr component */

    put_bits(pb, 8, 3); /* index */

    put_bits(pb, 4, 1); /* DC huffman table index */

    put_bits(pb, 4, lossless ? 0 : 1); /* AC huffman table index */



    put_bits(pb, 8, lossless ? avctx->prediction_method + 1 : 0); /* Ss (not used) */



    switch (avctx->codec_id) {

    case AV_CODEC_ID_MJPEG:  put_bits(pb, 8, 63); break; /* Se (not used) */

    case AV_CODEC_ID_LJPEG:  put_bits(pb, 8,  0); break; /* not used */

    default: assert(0);

    }



    put_bits(pb, 8, 0); /* Ah/Al (not used) */

}
