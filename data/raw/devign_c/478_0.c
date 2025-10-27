void avcodec_string(char *buf, int buf_size, AVCodecContext *enc, int encode)

{

    const char *codec_name;

    AVCodec *p;

    char buf1[32];

    int bitrate;

    AVRational display_aspect_ratio;



    if (encode)

        p = avcodec_find_encoder(enc->codec_id);

    else

        p = avcodec_find_decoder(enc->codec_id);



    if (p) {

        codec_name = p->name;

        if (!encode && enc->codec_id == CODEC_ID_MP3) {

            if (enc->sub_id == 2)

                codec_name = "mp2";

            else if (enc->sub_id == 1)

                codec_name = "mp1";

        }

    } else if (enc->codec_id == CODEC_ID_MPEG2TS) {

        /* fake mpeg2 transport stream codec (currently not

           registered) */

        codec_name = "mpeg2ts";

    } else if (enc->codec_name[0] != '\0') {

        codec_name = enc->codec_name;

    } else {

        /* output avi tags */

        if(   isprint(enc->codec_tag&0xFF) && isprint((enc->codec_tag>>8)&0xFF)

           && isprint((enc->codec_tag>>16)&0xFF) && isprint((enc->codec_tag>>24)&0xFF)){

            snprintf(buf1, sizeof(buf1), "%c%c%c%c / 0x%04X",

                     enc->codec_tag & 0xff,

                     (enc->codec_tag >> 8) & 0xff,

                     (enc->codec_tag >> 16) & 0xff,

                     (enc->codec_tag >> 24) & 0xff,

                      enc->codec_tag);

        } else {

            snprintf(buf1, sizeof(buf1), "0x%04x", enc->codec_tag);

        }

        codec_name = buf1;

    }



    switch(enc->codec_type) {

    case CODEC_TYPE_VIDEO:

        snprintf(buf, buf_size,

                 "Video: %s%s",

                 codec_name, enc->mb_decision ? " (hq)" : "");

        if (enc->pix_fmt != PIX_FMT_NONE) {

            snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", %s",

                     avcodec_get_pix_fmt_name(enc->pix_fmt));

        }

        if (enc->width) {

            snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", %dx%d",

                     enc->width, enc->height);

            if (enc->sample_aspect_ratio.num) {

                av_reduce(&display_aspect_ratio.num, &display_aspect_ratio.den,

                          enc->width*enc->sample_aspect_ratio.num,

                          enc->height*enc->sample_aspect_ratio.den,

                          1024*1024);

                snprintf(buf + strlen(buf), buf_size - strlen(buf),

                         " [PAR %d:%d DAR %d:%d]",

                         enc->sample_aspect_ratio.num, enc->sample_aspect_ratio.den,

                         display_aspect_ratio.num, display_aspect_ratio.den);

            }

            if(av_log_get_level() >= AV_LOG_DEBUG){

                int g= ff_gcd(enc->time_base.num, enc->time_base.den);

                snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", %d/%d",

                     enc->time_base.num/g, enc->time_base.den/g);

            }

        }

        if (encode) {

            snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", q=%d-%d", enc->qmin, enc->qmax);

        }

        bitrate = enc->bit_rate;

        break;

    case CODEC_TYPE_AUDIO:

        snprintf(buf, buf_size,

                 "Audio: %s",

                 codec_name);

        if (enc->sample_rate) {

            snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", %d Hz", enc->sample_rate);

        }

        av_strlcat(buf, ", ", buf_size);

        avcodec_get_channel_layout_string(buf + strlen(buf), buf_size - strlen(buf), enc->channels, enc->channel_layout);

        if (enc->sample_fmt != SAMPLE_FMT_NONE) {

            snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", %s", avcodec_get_sample_fmt_name(enc->sample_fmt));

        }



        /* for PCM codecs, compute bitrate directly */

        switch(enc->codec_id) {

        case CODEC_ID_PCM_F64BE:

        case CODEC_ID_PCM_F64LE:

            bitrate = enc->sample_rate * enc->channels * 64;

            break;

        case CODEC_ID_PCM_S32LE:

        case CODEC_ID_PCM_S32BE:

        case CODEC_ID_PCM_U32LE:

        case CODEC_ID_PCM_U32BE:

        case CODEC_ID_PCM_F32BE:

        case CODEC_ID_PCM_F32LE:

            bitrate = enc->sample_rate * enc->channels * 32;

            break;

        case CODEC_ID_PCM_S24LE:

        case CODEC_ID_PCM_S24BE:

        case CODEC_ID_PCM_U24LE:

        case CODEC_ID_PCM_U24BE:

        case CODEC_ID_PCM_S24DAUD:

            bitrate = enc->sample_rate * enc->channels * 24;

            break;

        case CODEC_ID_PCM_S16LE:

        case CODEC_ID_PCM_S16BE:

        case CODEC_ID_PCM_S16LE_PLANAR:

        case CODEC_ID_PCM_U16LE:

        case CODEC_ID_PCM_U16BE:

            bitrate = enc->sample_rate * enc->channels * 16;

            break;

        case CODEC_ID_PCM_S8:

        case CODEC_ID_PCM_U8:

        case CODEC_ID_PCM_ALAW:

        case CODEC_ID_PCM_MULAW:

        case CODEC_ID_PCM_ZORK:

            bitrate = enc->sample_rate * enc->channels * 8;

            break;

        default:

            bitrate = enc->bit_rate;

            break;

        }

        break;

    case CODEC_TYPE_DATA:

        snprintf(buf, buf_size, "Data: %s", codec_name);

        bitrate = enc->bit_rate;

        break;

    case CODEC_TYPE_SUBTITLE:

        snprintf(buf, buf_size, "Subtitle: %s", codec_name);

        bitrate = enc->bit_rate;

        break;

    case CODEC_TYPE_ATTACHMENT:

        snprintf(buf, buf_size, "Attachment: %s", codec_name);

        bitrate = enc->bit_rate;

        break;

    default:

        snprintf(buf, buf_size, "Invalid Codec type %d", enc->codec_type);

        return;

    }

    if (encode) {

        if (enc->flags & CODEC_FLAG_PASS1)

            snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", pass 1");

        if (enc->flags & CODEC_FLAG_PASS2)

            snprintf(buf + strlen(buf), buf_size - strlen(buf),

                     ", pass 2");

    }

    if (bitrate != 0) {

        snprintf(buf + strlen(buf), buf_size - strlen(buf),

                 ", %d kb/s", bitrate / 1000);

    }

}
