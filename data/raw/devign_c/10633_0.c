static int sdp_parse_rtpmap(AVCodecContext *codec, RTSPStream *rtsp_st, int payload_type, const char *p)

{

    char buf[256];

    int i;

    AVCodec *c;

    const char *c_name;



    /* Loop into AVRtpDynamicPayloadTypes[] and AVRtpPayloadTypes[] and

       see if we can handle this kind of payload */

    get_word_sep(buf, sizeof(buf), "/", &p);

    if (payload_type >= RTP_PT_PRIVATE) {

        RTPDynamicProtocolHandler *handler= RTPFirstDynamicPayloadHandler;

        while(handler) {

            if (!strcasecmp(buf, handler->enc_name) && (codec->codec_type == handler->codec_type)) {

                codec->codec_id = handler->codec_id;

                rtsp_st->dynamic_handler= handler;

                if(handler->open) {

                    rtsp_st->dynamic_protocol_context= handler->open();

                }

                break;

            }

            handler= handler->next;

        }

    } else {

        /* We are in a standard case ( from http://www.iana.org/assignments/rtp-parameters) */

        /* search into AVRtpPayloadTypes[] */

        codec->codec_id = ff_rtp_codec_id(buf, codec->codec_type);

    }



    c = avcodec_find_decoder(codec->codec_id);

    if (c && c->name)

        c_name = c->name;

    else

        c_name = (char *)NULL;



    if (c_name) {

        get_word_sep(buf, sizeof(buf), "/", &p);

        i = atoi(buf);

        switch (codec->codec_type) {

            case CODEC_TYPE_AUDIO:

                av_log(codec, AV_LOG_DEBUG, " audio codec set to : %s\n", c_name);

                codec->sample_rate = RTSP_DEFAULT_AUDIO_SAMPLERATE;

                codec->channels = RTSP_DEFAULT_NB_AUDIO_CHANNELS;

                if (i > 0) {

                    codec->sample_rate = i;

                    get_word_sep(buf, sizeof(buf), "/", &p);

                    i = atoi(buf);

                    if (i > 0)

                        codec->channels = i;

                    // TODO: there is a bug here; if it is a mono stream, and less than 22000Hz, faad upconverts to stereo and twice the

                    //  frequency.  No problem, but the sample rate is being set here by the sdp line.  Upcoming patch forthcoming. (rdm)

                }

                av_log(codec, AV_LOG_DEBUG, " audio samplerate set to : %i\n", codec->sample_rate);

                av_log(codec, AV_LOG_DEBUG, " audio channels set to : %i\n", codec->channels);

                break;

            case CODEC_TYPE_VIDEO:

                av_log(codec, AV_LOG_DEBUG, " video codec set to : %s\n", c_name);

                break;

            default:

                break;

        }

        return 0;

    }



    return -1;

}
