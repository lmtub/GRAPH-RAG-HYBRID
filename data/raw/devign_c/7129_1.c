static int pcm_read_header(AVFormatContext *s)

{

    PCMAudioDemuxerContext *s1 = s->priv_data;

    AVStream *st;

    uint8_t *mime_type = NULL;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);





    st->codecpar->codec_type  = AVMEDIA_TYPE_AUDIO;

    st->codecpar->codec_id    = s->iformat->raw_codec_id;

    st->codecpar->sample_rate = s1->sample_rate;

    st->codecpar->channels    = s1->channels;



    av_opt_get(s->pb, "mime_type", AV_OPT_SEARCH_CHILDREN, &mime_type);

    if (mime_type && s->iformat->mime_type) {

        int rate = 0, channels = 0;

        size_t len = strlen(s->iformat->mime_type);

        if (!strncmp(s->iformat->mime_type, mime_type, len)) {

            uint8_t *options = mime_type + len;

            len = strlen(mime_type);

            while (options < mime_type + len) {

                options = strstr(options, ";");

                if (!options++)

                    break;

                if (!rate)

                    sscanf(options, " rate=%d",     &rate);

                if (!channels)

                    sscanf(options, " channels=%d", &channels);

            }

            if (rate <= 0) {

                av_log(s, AV_LOG_ERROR,

                       "Invalid sample_rate found in mime_type \"%s\"\n",

                       mime_type);


                return AVERROR_INVALIDDATA;

            }

            st->codecpar->sample_rate = rate;

            if (channels > 0)

                st->codecpar->channels = channels;

        }

    }




    st->codecpar->bits_per_coded_sample =

        av_get_bits_per_sample(st->codecpar->codec_id);



    av_assert0(st->codecpar->bits_per_coded_sample > 0);



    st->codecpar->block_align =

        st->codecpar->bits_per_coded_sample * st->codecpar->channels / 8;



    avpriv_set_pts_info(st, 64, 1, st->codecpar->sample_rate);

    return 0;

}