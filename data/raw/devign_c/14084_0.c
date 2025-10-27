static av_cold int encode_end(AVCodecContext *avctx)

{

    LclEncContext *c = avctx->priv_data;



    av_freep(&avctx->extradata);

    deflateEnd(&c->zstream);



    av_frame_free(&avctx->coded_frame);



    return 0;

}
