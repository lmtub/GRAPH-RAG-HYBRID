int ff_rtp_get_payload_type(AVFormatContext *fmt,

                            AVCodecContext *codec, int idx)

{

    int i;

    AVOutputFormat *ofmt = fmt ? fmt->oformat : NULL;



    /* Was the payload type already specified for the RTP muxer? */

    if (ofmt && ofmt->priv_class && fmt->priv_data) {

        int64_t payload_type;

        if (av_opt_get_int(fmt->priv_data, "payload_type", 0, &payload_type) >= 0 &&

            payload_type >= 0)

            return (int)payload_type;

    }



    /* static payload type */

    for (i = 0; rtp_payload_types[i].pt >= 0; ++i)

        if (rtp_payload_types[i].codec_id == codec->codec_id) {

            if (codec->codec_id == AV_CODEC_ID_H263 && (!fmt ||

                !fmt->oformat->priv_class ||

                !av_opt_flag_is_set(fmt->priv_data, "rtpflags", "rfc2190")))

                continue;

            /* G722 has 8000 as nominal rate even if the sample rate is 16000,

             * see section 4.5.2 in RFC 3551. */

            if (codec->codec_id == AV_CODEC_ID_ADPCM_G722 &&

                codec->sample_rate == 16000 && codec->channels == 1)

                return rtp_payload_types[i].pt;

            if (codec->codec_type == AVMEDIA_TYPE_AUDIO &&

                ((rtp_payload_types[i].clock_rate > 0 &&

                  codec->sample_rate != rtp_payload_types[i].clock_rate) ||

                 (rtp_payload_types[i].audio_channels > 0 &&

                  codec->channels != rtp_payload_types[i].audio_channels)))

                continue;

            return rtp_payload_types[i].pt;

        }



    if (idx < 0)

        idx = codec->codec_type == AVMEDIA_TYPE_AUDIO;



    /* dynamic payload type */

    return RTP_PT_PRIVATE + idx;

}
