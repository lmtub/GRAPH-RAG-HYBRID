static av_cold int libgsm_init(AVCodecContext *avctx) {

    if (avctx->channels > 1) {

        av_log(avctx, AV_LOG_ERROR, "Mono required for GSM, got %d channels\n",

               avctx->channels);

        return -1;

    }



    if(avctx->codec->decode){

        if(!avctx->channels)

            avctx->channels= 1;



        if(!avctx->sample_rate)

            avctx->sample_rate= 8000;



        avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    }else{

        if (avctx->sample_rate != 8000) {

            av_log(avctx, AV_LOG_ERROR, "Sample rate 8000Hz required for GSM, got %dHz\n",

                avctx->sample_rate);

            if(avctx->strict_std_compliance > FF_COMPLIANCE_UNOFFICIAL)

                return -1;

        }

        if (avctx->bit_rate != 13000 /* Official */ &&

            avctx->bit_rate != 13200 /* Very common */ &&

            avctx->bit_rate != 0 /* Unknown; a.o. mov does not set bitrate when decoding */ ) {

            av_log(avctx, AV_LOG_ERROR, "Bitrate 13000bps required for GSM, got %dbps\n",

                avctx->bit_rate);

            if(avctx->strict_std_compliance > FF_COMPLIANCE_UNOFFICIAL)

                return -1;

        }

    }



    avctx->priv_data = gsm_create();



    switch(avctx->codec_id) {

    case CODEC_ID_GSM:

        avctx->frame_size = GSM_FRAME_SIZE;

        avctx->block_align = GSM_BLOCK_SIZE;

        break;

    case CODEC_ID_GSM_MS: {

        int one = 1;

        gsm_option(avctx->priv_data, GSM_OPT_WAV49, &one);

        avctx->frame_size = 2*GSM_FRAME_SIZE;

        avctx->block_align = GSM_MS_BLOCK_SIZE;

        }

    }



    avctx->coded_frame= avcodec_alloc_frame();

    avctx->coded_frame->key_frame= 1;



    return 0;

}
