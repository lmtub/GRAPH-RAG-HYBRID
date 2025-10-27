static av_cold int roq_dpcm_encode_init(AVCodecContext *avctx)

{

    ROQDPCMContext *context = avctx->priv_data;



    if (avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "Audio must be mono or stereo\n");

        return -1;

    }

    if (avctx->sample_rate != 22050) {

        av_log(avctx, AV_LOG_ERROR, "Audio must be 22050 Hz\n");

        return -1;

    }

    if (avctx->sample_fmt != AV_SAMPLE_FMT_S16) {

        av_log(avctx, AV_LOG_ERROR, "Audio must be signed 16-bit\n");

        return -1;

    }



    avctx->frame_size = ROQ_FIRST_FRAME_SIZE;



    context->lastSample[0] = context->lastSample[1] = 0;



    avctx->coded_frame= avcodec_alloc_frame();

    if (!avctx->coded_frame)

        return AVERROR(ENOMEM);



    return 0;

}
