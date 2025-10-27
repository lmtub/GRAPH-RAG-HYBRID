static av_cold int ljpeg_encode_init(AVCodecContext *avctx)

{

    LJpegEncContext *s = avctx->priv_data;



    if ((avctx->pix_fmt == AV_PIX_FMT_YUV420P ||

         avctx->pix_fmt == AV_PIX_FMT_YUV422P ||

         avctx->pix_fmt == AV_PIX_FMT_YUV444P ||

         avctx->color_range == AVCOL_RANGE_MPEG) &&

        avctx->strict_std_compliance > FF_COMPLIANCE_UNOFFICIAL) {

        av_log(avctx, AV_LOG_ERROR,

               "Limited range YUV is non-standard, set strict_std_compliance to "

               "at least unofficial to use it.\n");

        return AVERROR(EINVAL);

    }



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame)




    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;



    s->scratch = av_malloc_array(avctx->width + 1, sizeof(*s->scratch));

    if (!s->scratch)

        goto fail;



    ff_idctdsp_init(&s->idsp, avctx);

    ff_init_scantable(s->idsp.idct_permutation, &s->scantable,

                      ff_zigzag_direct);



    ff_mjpeg_init_hvsample(avctx, s->hsample, s->vsample);



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