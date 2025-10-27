static int create_stream(AVFormatContext *s)

{

    XCBGrabContext *c = s->priv_data;

    AVStream *st      = avformat_new_stream(s, NULL);

    xcb_get_geometry_cookie_t gc;

    xcb_get_geometry_reply_t *geo;

    int ret;



    if (!st)

        return AVERROR(ENOMEM);



    ret = av_parse_video_size(&c->width, &c->height, c->video_size);

    if (ret < 0)

        return ret;



    ret = av_parse_video_rate(&st->avg_frame_rate, c->framerate);

    if (ret < 0)

        return ret;



    avpriv_set_pts_info(st, 64, 1, 1000000);



    gc  = xcb_get_geometry(c->conn, c->screen->root);

    geo = xcb_get_geometry_reply(c->conn, gc, NULL);



    c->width      = FFMIN(geo->width, c->width);

    c->height     = FFMIN(geo->height, c->height);

    c->time_base  = (AVRational){ st->avg_frame_rate.den,

                                  st->avg_frame_rate.num };

    c->time_frame = av_gettime();



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id   = AV_CODEC_ID_RAWVIDEO;

    st->codec->width      = c->width;

    st->codec->height     = c->height;

    st->codec->time_base  = c->time_base;



    ret = pixfmt_from_pixmap_format(s, geo->depth, &st->codec->pix_fmt);



    free(geo);



    return ret;

}
