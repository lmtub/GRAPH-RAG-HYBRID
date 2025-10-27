static int hls_write_trailer(struct AVFormatContext *s)

{

    HLSContext *hls = s->priv_data;

    AVFormatContext *oc = hls->avf;



    av_write_trailer(oc);

    hls->size = avio_tell(hls->avf->pb) - hls->start_pos;

    avio_closep(&oc->pb);

    avformat_free_context(oc);

    av_free(hls->basename);

    hls_append_segment(hls, hls->duration, hls->start_pos, hls->size);

    hls_window(s, 1);



    hls_free_segments(hls);

    avio_close(hls->pb);

    return 0;

}
