static av_cold int ljpeg_encode_close(AVCodecContext *avctx)

{

    LJpegEncContext *s = avctx->priv_data;



    av_frame_free(&avctx->coded_frame);

    av_freep(&s->scratch);



    return 0;

}
