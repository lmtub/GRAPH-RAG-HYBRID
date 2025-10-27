static av_cold int cinvideo_decode_init(AVCodecContext *avctx)

{

    CinVideoContext *cin = avctx->priv_data;

    unsigned int i;



    cin->avctx = avctx;

    avctx->pix_fmt = AV_PIX_FMT_PAL8;



    cin->frame.data[0] = NULL;



    cin->bitmap_size = avctx->width * avctx->height;

    for (i = 0; i < 3; ++i) {

        cin->bitmap_table[i] = av_mallocz(cin->bitmap_size);

        if (!cin->bitmap_table[i])

            av_log(avctx, AV_LOG_ERROR, "Can't allocate bitmap buffers.\n");

    }



    return 0;

}
