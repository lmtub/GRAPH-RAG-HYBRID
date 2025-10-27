static void read_xing_toc(AVFormatContext *s, int64_t filesize, int64_t duration)

{

    int i;

    MP3DecContext *mp3 = s->priv_data;

    int fill_index = mp3->usetoc == 1 && duration > 0;



    if (!filesize &&

        !(filesize = avio_size(s->pb))) {

        av_log(s, AV_LOG_WARNING, "Cannot determine file size, skipping TOC table.\n");

        fill_index = 0;

    }



    for (i = 0; i < XING_TOC_COUNT; i++) {

        uint8_t b = avio_r8(s->pb);

        if (fill_index)

            av_add_index_entry(s->streams[0],

                           av_rescale(b, filesize, 256),

                           av_rescale(i, duration, XING_TOC_COUNT),

                           0, 0, AVINDEX_KEYFRAME);

    }

    if (fill_index)

        mp3->xing_toc = 1;

}
