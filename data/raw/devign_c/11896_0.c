static av_cold int flashsv_encode_end(AVCodecContext *avctx)

{

    FlashSVContext *s = avctx->priv_data;



    deflateEnd(&s->zstream);



    av_free(s->encbuffer);

    av_free(s->previous_frame);

    av_free(s->tmpblock);



    av_frame_free(&avctx->coded_frame);



    return 0;

}
