static int seg_write_trailer(struct AVFormatContext *s)

{

    SegmentContext *seg = s->priv_data;

    AVFormatContext *oc = seg->avf;

    int ret = 0;



    if (!oc)

        goto fail;



    if (!seg->write_header_trailer) {

        if ((ret = segment_end(oc, 0)) < 0)

            goto fail;

        open_null_ctx(&oc->pb);

        ret = av_write_trailer(oc);

        close_null_ctx(oc->pb);

    } else {

        ret = segment_end(oc, 1);

    }



    if (ret < 0)

        goto fail;



    if (seg->list && seg->list_type == LIST_HLS) {

        if ((ret = segment_hls_window(s, 1) < 0))

            goto fail;

    }



fail:

    avio_close(seg->pb);

    avformat_free_context(oc);

    return ret;

}
