static av_cold int seqvideo_decode_init(AVCodecContext *avctx)

{

    SeqVideoContext *seq = avctx->priv_data;



    seq->avctx = avctx;

    avctx->pix_fmt = AV_PIX_FMT_PAL8;



    seq->frame.data[0] = NULL;



    return 0;

}
