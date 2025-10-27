static int try_decode_frame(AVStream *st, AVPacket *avpkt)

{

    int16_t *samples;

    AVCodec *codec;

    int got_picture, data_size, ret=0;

    AVFrame picture;



    if(!st->codec->codec){

        codec = avcodec_find_decoder(st->codec->codec_id);

        if (!codec)

            return -1;

        ret = avcodec_open(st->codec, codec);

        if (ret < 0)

            return ret;

    }



    if(!has_codec_parameters(st->codec)){

        switch(st->codec->codec_type) {

        case CODEC_TYPE_VIDEO:


            ret = avcodec_decode_video2(st->codec, &picture,

                                        &got_picture, avpkt);

            break;

        case CODEC_TYPE_AUDIO:

            data_size = FFMAX(avpkt->size, AVCODEC_MAX_AUDIO_FRAME_SIZE);

            samples = av_malloc(data_size);

            if (!samples)

                goto fail;

            ret = avcodec_decode_audio3(st->codec, samples,

                                        &data_size, avpkt);

            av_free(samples);

            break;

        default:

            break;

        }

    }

 fail:

    return ret;

}