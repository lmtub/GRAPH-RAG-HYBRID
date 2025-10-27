static int img_read_packet(AVFormatContext *s1, AVPacket *pkt)

{

    VideoData *s = s1->priv_data;

    char filename[1024];

    int ret;

    ByteIOContext f1, *f;



    if (get_frame_filename(filename, sizeof(filename),

                           s->path, s->img_number) < 0)

        return -EIO;

    

    if (!s->is_pipe) {

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



    switch(s->img_fmt) {

    case IMGFMT_PGMYUV:

        ret = pgm_read(s, f, pkt->data, pkt->size, 1);

        break;

    case IMGFMT_PGM:

        ret = pgm_read(s, f, pkt->data, pkt->size, 0);

        break;

    case IMGFMT_YUV:

        ret = yuv_read(s, filename, pkt->data, pkt->size);

        break;

    case IMGFMT_PPM:

        ret = ppm_read(s, f, pkt->data, pkt->size);

        break;

    default:

        return -EIO;

    }

    

    if (!s->is_pipe) {

        url_fclose(f);

    }



    if (ret < 0) {

        av_free_packet(pkt);

        return -EIO; /* signal EOF */

    } else {

        s->img_number++;

        return 0;

    }

}
