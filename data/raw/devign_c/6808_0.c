static int tta_read_header(AVFormatContext *s)

{

    TTAContext *c = s->priv_data;

    AVStream *st;

    int i, channels, bps, samplerate;

    uint64_t framepos, start_offset;

    uint32_t datalen;



    if (!av_dict_get(s->metadata, "", NULL, AV_DICT_IGNORE_SUFFIX))

        ff_id3v1_read(s);



    start_offset = avio_tell(s->pb);

    if (avio_rl32(s->pb) != AV_RL32("TTA1"))

        return -1; // not tta file



    avio_skip(s->pb, 2); // FIXME: flags

    channels = avio_rl16(s->pb);

    bps = avio_rl16(s->pb);

    samplerate = avio_rl32(s->pb);

    if(samplerate <= 0 || samplerate > 1000000){

        av_log(s, AV_LOG_ERROR, "nonsense samplerate\n");

        return -1;

    }



    datalen = avio_rl32(s->pb);

    if (!datalen) {

        av_log(s, AV_LOG_ERROR, "invalid datalen\n");

        return AVERROR_INVALIDDATA;

    }



    avio_skip(s->pb, 4); // header crc



    c->frame_size      = samplerate * 256 / 245;

    c->last_frame_size = datalen % c->frame_size;

    if (!c->last_frame_size)

        c->last_frame_size = c->frame_size;

    c->totalframes = datalen / c->frame_size + (c->last_frame_size < c->frame_size);

    c->currentframe = 0;



    if(c->totalframes >= UINT_MAX/sizeof(uint32_t) || c->totalframes <= 0){

        av_log(s, AV_LOG_ERROR, "totalframes %d invalid\n", c->totalframes);

        return -1;

    }



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);



    avpriv_set_pts_info(st, 64, 1, samplerate);

    st->start_time = 0;

    st->duration = datalen;



    framepos = avio_tell(s->pb) + 4*c->totalframes + 4;



    for (i = 0; i < c->totalframes; i++) {

        uint32_t size = avio_rl32(s->pb);

        av_add_index_entry(st, framepos, i * c->frame_size, size, 0,

                           AVINDEX_KEYFRAME);

        framepos += size;

    }

    avio_skip(s->pb, 4); // seektable crc



    st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id = AV_CODEC_ID_TTA;

    st->codec->channels = channels;

    st->codec->sample_rate = samplerate;

    st->codec->bits_per_coded_sample = bps;



    st->codec->extradata_size = avio_tell(s->pb) - start_offset;

    if(st->codec->extradata_size+FF_INPUT_BUFFER_PADDING_SIZE <= (unsigned)st->codec->extradata_size){

        //this check is redundant as avio_read should fail

        av_log(s, AV_LOG_ERROR, "extradata_size too large\n");

        return -1;

    }

    st->codec->extradata = av_mallocz(st->codec->extradata_size+FF_INPUT_BUFFER_PADDING_SIZE);

    if (!st->codec->extradata) {

        st->codec->extradata_size = 0;

        return AVERROR(ENOMEM);

    }

    avio_seek(s->pb, start_offset, SEEK_SET);

    avio_read(s->pb, st->codec->extradata, st->codec->extradata_size);



    return 0;

}
