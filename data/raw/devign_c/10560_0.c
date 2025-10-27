static int read_seek(AVFormatContext *s, int stream_index, int64_t timestamp, int flags)

{

    BinkDemuxContext *bink = s->priv_data;

    AVStream *vst = s->streams[0];



    if (!s->pb->seekable)

        return -1;



    /* seek to the first frame */

    if (avio_seek(s->pb, vst->index_entries[0].pos, SEEK_SET) < 0)

        return -1;



    bink->video_pts = 0;

    memset(bink->audio_pts, 0, sizeof(bink->audio_pts));

    bink->current_track = -1;

    return 0;

}
