static void m4sl_cb(MpegTSFilter *filter, const uint8_t *section,

                    int section_len)

{

    MpegTSContext *ts = filter->u.section_filter.opaque;

    MpegTSSectionFilter *tssf = &filter->u.section_filter;

    SectionHeader h;

    const uint8_t *p, *p_end;

    AVIOContext pb;

    int mp4_descr_count = 0;

    Mp4Descr mp4_descr[MAX_MP4_DESCR_COUNT] = { { 0 } };

    int i, pid;

    AVFormatContext *s = ts->stream;



    p_end = section + section_len - 4;

    p = section;

    if (parse_section_header(&h, &p, p_end) < 0)

        return;

    if (h.tid != M4OD_TID)

        return;

    if (h.version == tssf->last_ver)

        return;

    tssf->last_ver = h.version;



    mp4_read_od(s, p, (unsigned) (p_end - p), mp4_descr, &mp4_descr_count,

                MAX_MP4_DESCR_COUNT);



    for (pid = 0; pid < NB_PID_MAX; pid++) {

        if (!ts->pids[pid])

            continue;

        for (i = 0; i < mp4_descr_count; i++) {

            PESContext *pes;

            AVStream *st;

            if (ts->pids[pid]->es_id != mp4_descr[i].es_id)

                continue;

            if (ts->pids[pid]->type != MPEGTS_PES) {

                av_log(s, AV_LOG_ERROR, "pid %x is not PES\n", pid);

                continue;

            }

            pes = ts->pids[pid]->u.pes_filter.opaque;

            st  = pes->st;

            if (!st)

                continue;



            pes->sl = mp4_descr[i].sl;



            ffio_init_context(&pb, mp4_descr[i].dec_config_descr,

                              mp4_descr[i].dec_config_descr_len, 0,

                              NULL, NULL, NULL, NULL);

            ff_mp4_read_dec_config_descr(s, st, &pb);

            if (st->codec->codec_id == AV_CODEC_ID_AAC &&

                st->codec->extradata_size > 0)

                st->need_parsing = 0;

            if (st->codec->codec_id == AV_CODEC_ID_H264 &&

                st->codec->extradata_size > 0)

                st->need_parsing = 0;



            if (st->codec->codec_id <= AV_CODEC_ID_NONE) {

                // do nothing

            } else if (st->codec->codec_id < AV_CODEC_ID_FIRST_AUDIO)

                st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

            else if (st->codec->codec_id < AV_CODEC_ID_FIRST_SUBTITLE)

                st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

            else if (st->codec->codec_id < AV_CODEC_ID_FIRST_UNKNOWN)

                st->codec->codec_type = AVMEDIA_TYPE_SUBTITLE;

        }

    }

    for (i = 0; i < mp4_descr_count; i++)

        av_free(mp4_descr[i].dec_config_descr);

}
