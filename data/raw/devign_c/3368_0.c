static int replaygain_export(AVStream *st,

                             const uint8_t *track_gain, const uint8_t *track_peak,

                             const uint8_t *album_gain, const uint8_t *album_peak)

{

    AVPacketSideData *sd, *tmp;

    AVReplayGain *replaygain;

    int32_t tg, ag;

    uint32_t tp, ap;



    tg = parse_value(track_gain, INT32_MIN);

    ag = parse_value(album_gain, INT32_MIN);

    tp = parse_value(track_peak, 0);

    ap = parse_value(album_peak, 0);



    if (tg == INT32_MIN && ag == INT32_MIN)

        return 0;



    replaygain = av_mallocz(sizeof(*replaygain));

    if (!replaygain)

        return AVERROR(ENOMEM);



    tmp = av_realloc_array(st->side_data, st->nb_side_data + 1, sizeof(*tmp));

    if (!tmp) {

        av_freep(&replaygain);

        return AVERROR(ENOMEM);

    }

    st->side_data = tmp;

    st->nb_side_data++;



    sd = &st->side_data[st->nb_side_data - 1];

    sd->type = AV_PKT_DATA_REPLAYGAIN;

    sd->data = (uint8_t*)replaygain;

    sd->size = sizeof(*replaygain);



    replaygain->track_gain = tg;

    replaygain->track_peak = tp;

    replaygain->album_gain = ag;

    replaygain->album_peak = ap;



    return 0;

}
