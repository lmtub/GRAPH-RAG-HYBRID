static int v4l2_read_packet(AVFormatContext *s1, AVPacket *pkt)

{

#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    struct video_data *s = s1->priv_data;

    AVFrame *frame = s1->streams[0]->codec->coded_frame;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

    int res;



    av_init_packet(pkt);

    if ((res = mmap_read_frame(s1, pkt)) < 0) {

        return res;

    }



#if FF_API_CODED_FRAME

FF_DISABLE_DEPRECATION_WARNINGS

    if (frame && s->interlaced) {

        frame->interlaced_frame = 1;

        frame->top_field_first = s->top_field_first;

    }

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    return pkt->size;

}
