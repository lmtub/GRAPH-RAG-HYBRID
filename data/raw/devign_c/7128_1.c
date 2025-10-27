static int nist_read_header(AVFormatContext *s)

{

    char buffer[32], coding[32] = "pcm", format[32] = "01";

    int bps = 0, be = 0;

    int32_t header_size;

    AVStream *st;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;



    ff_get_line(s->pb, buffer, sizeof(buffer));

    ff_get_line(s->pb, buffer, sizeof(buffer));

    sscanf(buffer, "%"SCNd32, &header_size);

    if (header_size <= 0)

        return AVERROR_INVALIDDATA;



    while (!url_feof(s->pb)) {

        ff_get_line(s->pb, buffer, sizeof(buffer));



        if (avio_tell(s->pb) >= header_size)

            return AVERROR_INVALIDDATA;



        if (!memcmp(buffer, "end_head", 8)) {

            if (!st->codec->bits_per_coded_sample)

                st->codec->bits_per_coded_sample = bps << 3;



            if (!av_strcasecmp(coding, "pcm")) {

                st->codec->codec_id = ff_get_pcm_codec_id(st->codec->bits_per_coded_sample,

                                                          0, be, 0xFFFF);

            } else if (!av_strcasecmp(coding, "alaw")) {

                st->codec->codec_id = AV_CODEC_ID_PCM_ALAW;

            } else if (!av_strcasecmp(coding, "ulaw") ||

                       !av_strcasecmp(coding, "mu-law")) {

                st->codec->codec_id = AV_CODEC_ID_PCM_MULAW;

            } else {

                avpriv_request_sample(s, "coding %s", coding);

            }



            avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);



            st->codec->block_align = st->codec->bits_per_coded_sample * st->codec->channels / 8;



            if (avio_tell(s->pb) > header_size)

                return AVERROR_INVALIDDATA;



            avio_skip(s->pb, header_size - avio_tell(s->pb));



            return 0;

        } else if (!memcmp(buffer, "channel_count", 13)) {

            sscanf(buffer, "%*s %*s %"SCNd32, &st->codec->channels);

        } else if (!memcmp(buffer, "sample_byte_format", 18)) {

            sscanf(buffer, "%*s %*s %31s", format);



            if (!av_strcasecmp(format, "01")) {

                be = 0;

            } else if (!av_strcasecmp(format, "10")) {

                be = 1;

            } else if (av_strcasecmp(format, "1")) {

                avpriv_request_sample(s, "sample byte format %s", format);

                return AVERROR_PATCHWELCOME;

            }

        } else if (!memcmp(buffer, "sample_coding", 13)) {

            sscanf(buffer, "%*s %*s %31s", coding);

        } else if (!memcmp(buffer, "sample_count", 12)) {

            sscanf(buffer, "%*s %*s %"SCNd64, &st->duration);

        } else if (!memcmp(buffer, "sample_n_bytes", 14)) {

            sscanf(buffer, "%*s %*s %"SCNd32, &bps);

        } else if (!memcmp(buffer, "sample_rate", 11)) {

            sscanf(buffer, "%*s %*s %"SCNd32, &st->codec->sample_rate);

        } else if (!memcmp(buffer, "sample_sig_bits", 15)) {

            sscanf(buffer, "%*s %*s %"SCNd32, &st->codec->bits_per_coded_sample);

        } else {

            char key[32], value[32];

            if (sscanf(buffer, "%31s %*s %31s", key, value) == 3) {

                av_dict_set(&s->metadata, key, value, AV_DICT_APPEND);

            } else {

                av_log(s, AV_LOG_ERROR, "Failed to parse '%s' as metadata\n", buffer);

            }

        }

    }



    return AVERROR_EOF;

}
