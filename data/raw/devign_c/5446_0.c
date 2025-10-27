static int img_read_header(AVFormatContext *s1, AVFormatParameters *ap)

{

    VideoData *s = s1->priv_data;

    int ret, first_index, last_index;

    char buf[1024];

    ByteIOContext pb1, *f = &pb1;

    AVStream *st;



    st = av_new_stream(s1, 0);

    if (!st) {

        av_free(s);

        return -ENOMEM;

    }



    if (ap && ap->image_format)

        s->img_fmt = ap->image_format;



    strcpy(s->path, s1->filename);

    s->img_number = 0;



    /* find format */

    if (s1->iformat->flags & AVFMT_NOFILE)

        s->is_pipe = 0;

    else

        s->is_pipe = 1;

        

    if (!ap || !ap->frame_rate) {

        st->codec.frame_rate      = 25;

        st->codec.frame_rate_base = 1;

    } else {

        st->codec.frame_rate      = ap->frame_rate;

        st->codec.frame_rate_base = ap->frame_rate_base;

    }

    

    if (!s->is_pipe) {

        if (find_image_range(&first_index, &last_index, s->path) < 0)

            goto fail;

        s->img_number = first_index;

        /* compute duration */

        st->start_time = 0;

        st->duration = ((int64_t)AV_TIME_BASE * 

                        (last_index - first_index + 1) * 

                        st->codec.frame_rate_base) / st->codec.frame_rate;

        if (get_frame_filename(buf, sizeof(buf), s->path, s->img_number) < 0)

            goto fail;

        if (url_fopen(f, buf, URL_RDONLY) < 0)

            goto fail;

    } else {

        f = &s1->pb;

    }

    

    ret = av_read_image(f, s1->filename, s->img_fmt, read_header_alloc_cb, s);

    if (ret < 0)

        goto fail1;



    if (!s->is_pipe) {

        url_fclose(f);

    } else {

        url_fseek(f, 0, SEEK_SET);

    }

    

    st->codec.codec_type = CODEC_TYPE_VIDEO;

    st->codec.codec_id = CODEC_ID_RAWVIDEO;

    st->codec.width = s->width;

    st->codec.height = s->height;

    st->codec.pix_fmt = s->pix_fmt;

    s->img_size = avpicture_get_size(s->pix_fmt, s->width, s->height);



    return 0;

 fail1:

    if (!s->is_pipe)

        url_fclose(f);

 fail:

    av_free(s);

    return -EIO;

}
