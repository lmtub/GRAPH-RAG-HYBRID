static int img_read_packet(AVFormatContext *s1, AVPacket *pkt)

{

    VideoData *s = s1->priv_data;

    char filename[1024];

    int ret;

    ByteIOContext f1, *f;



    if (!s->is_pipe) {

        if (get_frame_filename(filename, sizeof(filename),

                               s->path, s->img_number) < 0)

            return -EIO;

        f = &f1;

        if (url_fopen(f, filename, URL_RDONLY) < 0)

            return -EIO;

    } else {

        f = &s1->pb;

        if (url_feof(f))

            return -EIO;

    }



    av_new_packet(pkt, s->img_size);

    pkt->stream_index = 0;



    s->ptr = pkt->data;

    ret = av_read_image(f, filename, s->img_fmt, read_packet_alloc_cb, s);

    if (!s->is_pipe) {

        url_fclose(f);

    }



    if (ret < 0) {

        av_free_packet(pkt);

        return -EIO; /* signal EOF */

    } else {

        pkt->pts = av_rescale((int64_t)s->img_number * s1->streams[0]->codec.frame_rate_base, s1->pts_den, s1->streams[0]->codec.frame_rate) / s1->pts_num;

        s->img_number++;

        return 0;

    }

}
