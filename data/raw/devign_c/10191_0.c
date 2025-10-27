static int write_abst(AVFormatContext *s, OutputStream *os, int final)

{

    HDSContext *c = s->priv_data;

    AVIOContext *out;

    char filename[1024], temp_filename[1024];

    int i, ret;

    int64_t asrt_pos, afrt_pos;

    int start = 0, fragments;

    int index = s->streams[os->first_stream]->id;

    int64_t cur_media_time = 0;

    if (c->window_size)

        start = FFMAX(os->nb_fragments - c->window_size, 0);

    fragments = os->nb_fragments - start;

    if (final)

        cur_media_time = os->last_ts;

    else if (os->nb_fragments)

        cur_media_time = os->fragments[os->nb_fragments - 1]->start_time;



    snprintf(filename, sizeof(filename),

             "%s/stream%d.abst", s->filename, index);

    snprintf(temp_filename, sizeof(temp_filename),

             "%s/stream%d.abst.tmp", s->filename, index);

    ret = avio_open2(&out, temp_filename, AVIO_FLAG_WRITE,

                     &s->interrupt_callback, NULL);

    if (ret < 0) {

        av_log(s, AV_LOG_ERROR, "Unable to open %s for writing\n", temp_filename);

        return ret;

    }

    avio_wb32(out, 0); // abst size

    avio_wl32(out, MKTAG('a','b','s','t'));

    avio_wb32(out, 0); // version + flags

    avio_wb32(out, os->fragment_index - 1); // BootstrapinfoVersion

    avio_w8(out, final ? 0 : 0x20); // profile, live, update

    avio_wb32(out, 1000); // timescale

    avio_wb64(out, cur_media_time);

    avio_wb64(out, 0); // SmpteTimeCodeOffset

    avio_w8(out, 0); // MovieIdentifer (null string)

    avio_w8(out, 0); // ServerEntryCount

    avio_w8(out, 0); // QualityEntryCount

    avio_w8(out, 0); // DrmData (null string)

    avio_w8(out, 0); // MetaData (null string)

    avio_w8(out, 1); // SegmentRunTableCount

    asrt_pos = avio_tell(out);

    avio_wb32(out, 0); // asrt size

    avio_wl32(out, MKTAG('a','s','r','t'));

    avio_wb32(out, 0); // version + flags

    avio_w8(out, 0); // QualityEntryCount

    avio_wb32(out, 1); // SegmentRunEntryCount

    avio_wb32(out, 1); // FirstSegment

    avio_wb32(out, final ? (os->fragment_index - 1) : 0xffffffff); // FragmentsPerSegment

    update_size(out, asrt_pos);

    avio_w8(out, 1); // FragmentRunTableCount

    afrt_pos = avio_tell(out);

    avio_wb32(out, 0); // afrt size

    avio_wl32(out, MKTAG('a','f','r','t'));

    avio_wb32(out, 0); // version + flags

    avio_wb32(out, 1000); // timescale

    avio_w8(out, 0); // QualityEntryCount

    avio_wb32(out, fragments); // FragmentRunEntryCount

    for (i = start; i < os->nb_fragments; i++) {

        avio_wb32(out, os->fragments[i]->n);

        avio_wb64(out, os->fragments[i]->start_time);

        avio_wb32(out, os->fragments[i]->duration);

    }

    update_size(out, afrt_pos);

    update_size(out, 0);

    avio_close(out);

    return ff_rename(temp_filename, filename);

}
