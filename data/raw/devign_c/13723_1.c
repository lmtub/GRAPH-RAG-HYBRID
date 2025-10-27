ogm_dshow_header(AVFormatContext *s, int idx)
{
    struct ogg *ogg = s->priv_data;
    struct ogg_stream *os = ogg->streams + idx;
    AVStream *st = s->streams[idx];
    uint8_t *p = os->buf + os->pstart;
    uint32_t t;
    if(!(*p & 1))
        return 0;
    if(*p != 1)
        return 1;
    t = AV_RL32(p + 96);
    if(t == 0x05589f80){
        st->codec->codec_type = AVMEDIA_TYPE_VIDEO;
        st->codec->codec_id = ff_codec_get_id(ff_codec_bmp_tags, AV_RL32(p + 68));
        avpriv_set_pts_info(st, 64, AV_RL64(p + 164), 10000000);
        st->codec->width = AV_RL32(p + 176);
        st->codec->height = AV_RL32(p + 180);
    } else if(t == 0x05589f81){
        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;
        st->codec->codec_id = ff_codec_get_id(ff_codec_wav_tags, AV_RL16(p + 124));
        st->codec->channels = AV_RL16(p + 126);
        st->codec->sample_rate = AV_RL32(p + 128);
        st->codec->bit_rate = AV_RL32(p + 132) * 8;
    }
    return 1;
}