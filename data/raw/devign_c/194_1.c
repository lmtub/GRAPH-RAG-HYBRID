static int webvtt_read_header(AVFormatContext *s)

{

    WebVTTContext *webvtt = s->priv_data;

    AVBPrint header, cue;

    int res = 0;

    AVStream *st = avformat_new_stream(s, NULL);



    if (!st)

        return AVERROR(ENOMEM);

    avpriv_set_pts_info(st, 64, 1, 1000);

    st->codec->codec_type = AVMEDIA_TYPE_SUBTITLE;

    st->codec->codec_id   = AV_CODEC_ID_WEBVTT;

    st->disposition |= webvtt->kind;



    av_bprint_init(&header, 0, AV_BPRINT_SIZE_UNLIMITED);

    av_bprint_init(&cue,    0, AV_BPRINT_SIZE_UNLIMITED);



    for (;;) {

        int i;

        int64_t pos;

        AVPacket *sub;

        const char *p, *identifier, *settings;

        int identifier_len, settings_len;

        int64_t ts_start, ts_end;



        ff_subtitles_read_chunk(s->pb, &cue);



        if (!cue.len)

            break;



        p = identifier = cue.str;

        pos = avio_tell(s->pb);



        /* ignore header chunk */

        if (!strncmp(p, "\xEF\xBB\xBFWEBVTT", 9) ||

            !strncmp(p, "WEBVTT", 6))

            continue;



        /* optional cue identifier (can be a number like in SRT or some kind of

         * chaptering id) */

        for (i = 0; p[i] && p[i] != '\n' && p[i] != '\r'; i++) {

            if (!strncmp(p + i, "-->", 3)) {

                identifier = NULL;

                break;

            }

        }

        if (!identifier)

            identifier_len = 0;

        else {

            identifier_len = strcspn(p, "\r\n");

            p += identifier_len;

            if (*p == '\r')

                p++;

            if (*p == '\n')

                p++;

        }



        /* cue timestamps */

        if ((ts_start = read_ts(p)) == AV_NOPTS_VALUE)

            break;

        if (!(p = strstr(p, "-->")))

            break;

        p += 3;

        do p++; while (*p == ' ' || *p == '\t');

        if ((ts_end = read_ts(p)) == AV_NOPTS_VALUE)

            break;



        /* optional cue settings */

        p += strcspn(p, "\n\t ");

        while (*p == '\t' || *p == ' ')

            p++;

        settings = p;

        settings_len = strcspn(p, "\r\n");

        p += settings_len;

        if (*p == '\r')

            p++;

        if (*p == '\n')

            p++;



        /* create packet */

        sub = ff_subtitles_queue_insert(&webvtt->q, p, strlen(p), 0);

        if (!sub) {

            res = AVERROR(ENOMEM);

            goto end;

        }

        sub->pos = pos;

        sub->pts = ts_start;

        sub->duration = ts_end - ts_start;



#define SET_SIDE_DATA(name, type) do {                                  \

    if (name##_len) {                                                   \

        uint8_t *buf = av_packet_new_side_data(sub, type, name##_len);  \

        if (!buf) {                                                     \

            res = AVERROR(ENOMEM);                                      \

            goto end;                                                   \

        }                                                               \

        memcpy(buf, name, name##_len);                                  \

    }                                                                   \

} while (0)



        SET_SIDE_DATA(identifier, AV_PKT_DATA_WEBVTT_IDENTIFIER);

        SET_SIDE_DATA(settings,   AV_PKT_DATA_WEBVTT_SETTINGS);

    }



    ff_subtitles_queue_finalize(&webvtt->q);



end:

    av_bprint_finalize(&cue,    NULL);

    av_bprint_finalize(&header, NULL);

    return res;

}
