int attribute_align_arg avcodec_decode_audio2(AVCodecContext *avctx, int16_t *samples,

                         int *frame_size_ptr,

                         uint8_t *buf, int buf_size)

{

    int ret;



    if((avctx->codec->capabilities & CODEC_CAP_DELAY) || buf_size){

        //FIXME remove the check below _after_ ensuring that all audio check that the available space is enough

        if(*frame_size_ptr < AVCODEC_MAX_AUDIO_FRAME_SIZE){

            av_log(avctx, AV_LOG_ERROR, "buffer smaller than AVCODEC_MAX_AUDIO_FRAME_SIZE\n");

            return -1;

        }

        if(*frame_size_ptr < FF_MIN_BUFFER_SIZE ||

        *frame_size_ptr < avctx->channels * avctx->frame_size * sizeof(int16_t) ||

        *frame_size_ptr < buf_size){

            av_log(avctx, AV_LOG_ERROR, "buffer %d too small\n", *frame_size_ptr);

            return -1;

        }



        ret = avctx->codec->decode(avctx, samples, frame_size_ptr,

                                buf, buf_size);

        avctx->frame_number++;

    }else{

        ret= 0;

        *frame_size_ptr=0;

    }

    return ret;

}
