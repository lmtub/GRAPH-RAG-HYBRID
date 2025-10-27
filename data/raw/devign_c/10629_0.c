static int get_audio_flags(AVCodecContext *enc){

    int flags = (enc->bits_per_coded_sample == 16) ? FLV_SAMPLESSIZE_16BIT : FLV_SAMPLESSIZE_8BIT;



    if (enc->codec_id == CODEC_ID_AAC) // specs force these parameters

        return FLV_CODECID_AAC | FLV_SAMPLERATE_44100HZ | FLV_SAMPLESSIZE_16BIT | FLV_STEREO;

    else if (enc->codec_id == CODEC_ID_SPEEX) {

        if (enc->sample_rate != 16000) {

            av_log(enc, AV_LOG_ERROR, "flv only supports wideband (16kHz) Speex audio\n");

            return -1;

        }

        if (enc->channels != 1) {

            av_log(enc, AV_LOG_ERROR, "flv only supports mono Speex audio\n");

            return -1;

        }

        if (enc->frame_size / 320 > 8) {

            av_log(enc, AV_LOG_WARNING, "Warning: Speex stream has more than "

                                        "8 frames per packet. Adobe Flash "

                                        "Player cannot handle this!\n");

        }

        return FLV_CODECID_SPEEX | FLV_SAMPLERATE_11025HZ | FLV_SAMPLESSIZE_16BIT;

    } else {

    switch (enc->sample_rate) {

        case    44100:

            flags |= FLV_SAMPLERATE_44100HZ;

            break;

        case    22050:

            flags |= FLV_SAMPLERATE_22050HZ;

            break;

        case    11025:

            flags |= FLV_SAMPLERATE_11025HZ;

            break;

        case     8000: //nellymoser only

        case     5512: //not mp3

            if(enc->codec_id != CODEC_ID_MP3){

                flags |= FLV_SAMPLERATE_SPECIAL;

                break;

            }

        default:

            av_log(enc, AV_LOG_ERROR, "flv does not support that sample rate, choose from (44100, 22050, 11025).\n");

            return -1;

    }

    }



    if (enc->channels > 1) {

        flags |= FLV_STEREO;

    }



    switch(enc->codec_id){

    case CODEC_ID_MP3:

        flags |= FLV_CODECID_MP3    | FLV_SAMPLESSIZE_16BIT;

        break;

    case CODEC_ID_PCM_U8:

        flags |= FLV_CODECID_PCM    | FLV_SAMPLESSIZE_8BIT;

        break;

    case CODEC_ID_PCM_S16BE:

        flags |= FLV_CODECID_PCM    | FLV_SAMPLESSIZE_16BIT;

        break;

    case CODEC_ID_PCM_S16LE:

        flags |= FLV_CODECID_PCM_LE | FLV_SAMPLESSIZE_16BIT;

        break;

    case CODEC_ID_ADPCM_SWF:

        flags |= FLV_CODECID_ADPCM | FLV_SAMPLESSIZE_16BIT;

        break;

    case CODEC_ID_NELLYMOSER:

        if (enc->sample_rate == 8000) {

            flags |= FLV_CODECID_NELLYMOSER_8KHZ_MONO | FLV_SAMPLESSIZE_16BIT;

        } else {

            flags |= FLV_CODECID_NELLYMOSER | FLV_SAMPLESSIZE_16BIT;

        }

        break;

    case 0:

        flags |= enc->codec_tag<<4;

        break;

    default:

        av_log(enc, AV_LOG_ERROR, "codec not compatible with flv\n");

        return -1;

    }



    return flags;

}
