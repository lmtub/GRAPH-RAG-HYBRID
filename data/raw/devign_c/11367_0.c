static av_cold int ljpeg_encode_init(AVCodecContext *avctx)

{

    LJpegEncContext *s = avctx->priv_data;

    int chroma_v_shift, chroma_h_shift;



    if ((avctx->pix_fmt == AV_PIX_FMT_YUV420P ||

         avctx->pix_fmt == AV_PIX_FMT_YUV422P ||

         avctx->pix_fmt == AV_PIX_FMT_YUV444P) &&

        avctx->strict_std_compliance > FF_COMPLIANCE_UNOFFICIAL) {

        av_log(avctx, AV_LOG_ERROR,

               "Limited range YUV is non-standard, set strict_std_compliance to "

               "at least unofficial to use it.\n");

        return AVERROR(EINVAL);

    }



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;



    s->scratch = av_malloc_array(avctx->width + 1, sizeof(*s->scratch));



    ff_dsputil_init(&s->dsp, avctx);

    ff_init_scantable(s->dsp.idct_permutation, &s->scantable, ff_zigzag_direct);



    av_pix_fmt_get_chroma_sub_sample(avctx->pix_fmt, &chroma_h_shift,

                                     &chroma_v_shift);



    if (avctx->pix_fmt   == AV_PIX_FMT_BGR24) {

        s->vsample[0] = s->hsample[0] =

        s->vsample[1] = s->hsample[1] =

        s->vsample[2] = s->hsample[2] = 1;

    } else {

        s->vsample[0] = 2;

        s->vsample[1] = 2 >> chroma_v_shift;

        s->vsample[2] = 2 >> chroma_v_shift;

        s->hsample[0] = 2;

        s->hsample[1] = 2 >> chroma_h_shift;

        s->hsample[2] = 2 >> chroma_h_shift;

    }



    ff_mjpeg_build_huffman_codes(s->huff_size_dc_luminance,

                                 s->huff_code_dc_luminance,

                                 avpriv_mjpeg_bits_dc_luminance,

                                 avpriv_mjpeg_val_dc);

    ff_mjpeg_build_huffman_codes(s->huff_size_dc_chrominance,

                                 s->huff_code_dc_chrominance,

                                 avpriv_mjpeg_bits_dc_chrominance,

                                 avpriv_mjpeg_val_dc);



    return 0;

}
