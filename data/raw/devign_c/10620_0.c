static int segment_hls_window(AVFormatContext *s, int last)

{

    SegmentContext *seg = s->priv_data;

    int i, ret = 0;

    char buf[1024];



    if ((ret = avio_open2(&seg->pb, seg->list, AVIO_FLAG_WRITE,

                              &s->interrupt_callback, NULL)) < 0)

        goto fail;



    avio_printf(seg->pb, "#EXTM3U\n");

    avio_printf(seg->pb, "#EXT-X-VERSION:3\n");

    avio_printf(seg->pb, "#EXT-X-TARGETDURATION:%d\n", (int)seg->time);

    avio_printf(seg->pb, "#EXT-X-MEDIA-SEQUENCE:%d\n",

                FFMAX(0, seg->number - seg->size));



    av_log(s, AV_LOG_VERBOSE, "EXT-X-MEDIA-SEQUENCE:%d\n",

           FFMAX(0, seg->number - seg->size));



    for (i = FFMAX(0, seg->number - seg->size);

         i < seg->number; i++) {

        avio_printf(seg->pb, "#EXTINF:%d,\n", (int)seg->time);

        if (seg->entry_prefix) {

            avio_printf(seg->pb, "%s", seg->entry_prefix);

        }

        ret = av_get_frame_filename(buf, sizeof(buf), s->filename, i);

        if (ret < 0) {

            ret = AVERROR(EINVAL);

            goto fail;

        }

        avio_printf(seg->pb, "%s\n", buf);

    }



    if (last)

        avio_printf(seg->pb, "#EXT-X-ENDLIST\n");

fail:

    avio_closep(&seg->pb);

    return ret;

}
