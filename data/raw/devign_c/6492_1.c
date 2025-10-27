static av_cold int pcm_encode_init(AVCodecContext *avctx)

{

    avctx->frame_size = 0;

    switch(avctx->codec->id) {

    case CODEC_ID_PCM_ALAW:

        pcm_alaw_tableinit();

        break;

    case CODEC_ID_PCM_MULAW:

        pcm_ulaw_tableinit();

        break;

    default:

        break;

    }



    avctx->bits_per_coded_sample = av_get_bits_per_sample(avctx->codec->id);

    avctx->block_align = avctx->channels * avctx->bits_per_coded_sample/8;

    avctx->coded_frame= avcodec_alloc_frame();





    return 0;

}