static av_cold int qtrle_encode_init(AVCodecContext *avctx)

{

    QtrleEncContext *s = avctx->priv_data;



    if (av_image_check_size(avctx->width, avctx->height, 0, avctx) < 0) {

        return -1;

    }

    s->avctx=avctx;



    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_RGB555BE:

        s->pixel_size = 2;

        break;

    case AV_PIX_FMT_RGB24:

        s->pixel_size = 3;

        break;

    case AV_PIX_FMT_ARGB:

        s->pixel_size = 4;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unsupported colorspace.\n");

        break;

    }

    avctx->bits_per_coded_sample = s->pixel_size*8;



    s->rlecode_table = av_mallocz(s->avctx->width);

    s->skip_table    = av_mallocz(s->avctx->width);

    s->length_table  = av_mallocz((s->avctx->width + 1)*sizeof(int));

    if (!s->skip_table || !s->length_table || !s->rlecode_table) {

        av_log(avctx, AV_LOG_ERROR, "Error allocating memory.\n");

        return -1;

    }

    if (avpicture_alloc(&s->previous_frame, avctx->pix_fmt, avctx->width, avctx->height) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error allocating picture\n");

        return -1;

    }



    s->max_buf_size = s->avctx->width*s->avctx->height*s->pixel_size*2 /* image base material */

                      + 15                                           /* header + footer */

                      + s->avctx->height*2                           /* skip code+rle end */

                      + s->avctx->width/MAX_RLE_BULK + 1             /* rle codes */;



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame) {

        qtrle_encode_end(avctx);

        return AVERROR(ENOMEM);

    }



    return 0;

}
