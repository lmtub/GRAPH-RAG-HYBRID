static int parse_dsd_prop(AVFormatContext *s, AVStream *st, uint64_t eof)

{

    AVIOContext *pb = s->pb;

    char abss[24];

    int hour, min, sec, i, ret, config;

    int dsd_layout[6];

    ID3v2ExtraMeta *id3v2_extra_meta;



    while (avio_tell(pb) + 12 <= eof) {

        uint32_t tag      = avio_rl32(pb);

        uint64_t size     = avio_rb64(pb);

        uint64_t orig_pos = avio_tell(pb);



        switch(tag) {

        case MKTAG('A','B','S','S'):

            if (size < 8)

                return AVERROR_INVALIDDATA;

            hour = avio_rb16(pb);

            min  = avio_r8(pb);

            sec  = avio_r8(pb);

            snprintf(abss, sizeof(abss), "%02dh:%02dm:%02ds:%d", hour, min, sec, avio_rb32(pb));

            av_dict_set(&st->metadata, "absolute_start_time", abss, 0);

            break;



        case MKTAG('C','H','N','L'):

            if (size < 2)

                return AVERROR_INVALIDDATA;

            st->codecpar->channels       = avio_rb16(pb);

            if (size < 2 + st->codecpar->channels * 4)

                return AVERROR_INVALIDDATA;

            st->codecpar->channel_layout = 0;

            if (st->codecpar->channels > FF_ARRAY_ELEMS(dsd_layout)) {

                avpriv_request_sample(s, "channel layout");

                break;

            }

            for (i = 0; i < st->codecpar->channels; i++)

                dsd_layout[i] = avio_rl32(pb);

            for (i = 0; i < FF_ARRAY_ELEMS(dsd_channel_layout); i++) {

                const DSDLayoutDesc * d = &dsd_channel_layout[i];

                if (av_get_channel_layout_nb_channels(d->layout) == st->codecpar->channels &&

                    !memcmp(d->dsd_layout, dsd_layout, st->codecpar->channels * sizeof(uint32_t))) {

                    st->codecpar->channel_layout = d->layout;

                    break;

                }

            }

            break;



        case MKTAG('C','M','P','R'):

            if (size < 4)

                return AVERROR_INVALIDDATA;

            tag = avio_rl32(pb);

            st->codecpar->codec_id = ff_codec_get_id(dsd_codec_tags, tag);

            if (!st->codecpar->codec_id) {

                av_log(s, AV_LOG_ERROR, "'%c%c%c%c' compression is not supported\n",

                    tag&0xFF, (tag>>8)&0xFF, (tag>>16)&0xFF, (tag>>24)&0xFF);

                return AVERROR_PATCHWELCOME;

            }

            break;



        case MKTAG('F','S',' ',' '):

            if (size < 4)

                return AVERROR_INVALIDDATA;

            st->codecpar->sample_rate = avio_rb32(pb) / 8;

            break;



        case MKTAG('I','D','3',' '):

            id3v2_extra_meta = NULL;

            ff_id3v2_read(s, ID3v2_DEFAULT_MAGIC, &id3v2_extra_meta, size);

            if (id3v2_extra_meta) {

                if ((ret = ff_id3v2_parse_apic(s, &id3v2_extra_meta)) < 0) {

                    ff_id3v2_free_extra_meta(&id3v2_extra_meta);

                    return ret;

                }

                ff_id3v2_free_extra_meta(&id3v2_extra_meta);

            }



            if (size < avio_tell(pb) - orig_pos) {

                av_log(s, AV_LOG_ERROR, "id3 exceeds chunk size\n");

                return AVERROR_INVALIDDATA;

            }

            break;



        case MKTAG('L','S','C','O'):

            if (size < 2)

                return AVERROR_INVALIDDATA;

            config = avio_rb16(pb);

            if (config != 0xFFFF) {

                if (config < FF_ARRAY_ELEMS(dsd_loudspeaker_config))

                    st->codecpar->channel_layout = dsd_loudspeaker_config[config];

                if (!st->codecpar->channel_layout)

                    avpriv_request_sample(s, "loudspeaker configuration %d", config);

            }

            break;

        }



        avio_skip(pb, size - (avio_tell(pb) - orig_pos) + (size & 1));

    }



    return 0;

}
