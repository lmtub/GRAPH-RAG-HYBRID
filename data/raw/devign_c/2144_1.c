static int ircam_read_header(AVFormatContext *s)

{

    uint32_t magic, sample_rate, channels, tag;

    const AVCodecTag *tags;

    int le = -1, i;

    AVStream *st;



    magic = avio_rl32(s->pb);

    for (i = 0; i < 7; i++) {

        if (magic == table[i].magic) {

            le = table[i].is_le;

            break;

        }

    }



    if (le == 1) {

        sample_rate = av_int2float(avio_rl32(s->pb));

        channels    = avio_rl32(s->pb);

        tag         = avio_rl32(s->pb);

        tags        = ff_codec_ircam_le_tags;

    } else if (le == 0) {

        sample_rate = av_int2float(avio_rb32(s->pb));

        channels    = avio_rb32(s->pb);

        tag         = avio_rb32(s->pb);

        tags        = ff_codec_ircam_be_tags;

    } else {

        return AVERROR_INVALIDDATA;

    }



    if (!channels || !sample_rate)

        return AVERROR_INVALIDDATA;



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    st->codecpar->codec_type  = AVMEDIA_TYPE_AUDIO;

    st->codecpar->channels    = channels;



    st->codecpar->sample_rate = sample_rate;



    st->codecpar->codec_id = ff_codec_get_id(tags, tag);

    if (st->codecpar->codec_id == AV_CODEC_ID_NONE) {

        av_log(s, AV_LOG_ERROR, "unknown tag %X\n", tag);

        return AVERROR_INVALIDDATA;

    }



    st->codecpar->bits_per_coded_sample = av_get_bits_per_sample(st->codecpar->codec_id);

    st->codecpar->block_align = st->codecpar->bits_per_coded_sample * st->codecpar->channels / 8;

    avpriv_set_pts_info(st, 64, 1, st->codecpar->sample_rate);

    avio_skip(s->pb, 1008);



    return 0;

}