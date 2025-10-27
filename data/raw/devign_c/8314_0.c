av_cold int ff_mjpeg_decode_init(AVCodecContext *avctx)

{

    MJpegDecodeContext *s = avctx->priv_data;

    int ret;



    if (!s->picture_ptr) {

        s->picture = av_frame_alloc();

        if (!s->picture)

            return AVERROR(ENOMEM);

        s->picture_ptr = s->picture;

    }



    s->avctx = avctx;

    ff_blockdsp_init(&s->bdsp, avctx);

    ff_hpeldsp_init(&s->hdsp, avctx->flags);

    ff_idctdsp_init(&s->idsp, avctx);

    ff_init_scantable(s->idsp.idct_permutation, &s->scantable,

                      ff_zigzag_direct);

    s->buffer_size   = 0;

    s->buffer        = NULL;

    s->start_code    = -1;

    s->first_picture = 1;

    s->org_height    = avctx->coded_height;

    avctx->chroma_sample_location = AVCHROMA_LOC_CENTER;

    avctx->colorspace = AVCOL_SPC_BT470BG;



    if ((ret = build_basic_mjpeg_vlc(s)) < 0)

        return ret;



    if (s->extern_huff) {

        av_log(avctx, AV_LOG_INFO, "mjpeg: using external huffman table\n");

        if ((ret = init_get_bits(&s->gb, avctx->extradata, avctx->extradata_size * 8)) < 0)

            return ret;

        if ((ret = ff_mjpeg_decode_dht(s))) {

            av_log(avctx, AV_LOG_ERROR,

                   "mjpeg: error using external huffman table\n");

            return ret;

        }

    }

    if (avctx->field_order == AV_FIELD_BB) { /* quicktime icefloe 019 */

        s->interlace_polarity = 1;           /* bottom field first */

        av_log(avctx, AV_LOG_DEBUG, "mjpeg bottom field first\n");

    }

    if (avctx->codec->id == AV_CODEC_ID_AMV)

        s->flipped = 1;



    return 0;

}
