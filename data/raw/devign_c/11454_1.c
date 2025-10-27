static av_cold int cinvideo_decode_end(AVCodecContext *avctx)

{

    CinVideoContext *cin = avctx->priv_data;

    int i;



    if (cin->frame.data[0])

        avctx->release_buffer(avctx, &cin->frame);



    for (i = 0; i < 3; ++i)

        av_free(cin->bitmap_table[i]);



    return 0;

}
