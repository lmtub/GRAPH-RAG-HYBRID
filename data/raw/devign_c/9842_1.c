static int img_write_packet(AVFormatContext *s, int stream_index,

                            UINT8 *buf, int size)

{

    VideoData *img = s->priv_data;

    AVStream *st = s->streams[stream_index];

    ByteIOContext pb1, *pb;

    AVPicture picture;

    int width, height, ret, size1;

    char filename[1024];



    width = st->codec.width;

    height = st->codec.height;



    switch(st->codec.pix_fmt) {

    case PIX_FMT_YUV420P:

        size1 = (width * height * 3) / 2;

        if (size != size1)

            return -EIO;

        

        picture.data[0] = buf;

        picture.data[1] = picture.data[0] + width * height;

        picture.data[2] = picture.data[1] + (width * height) / 4;

        picture.linesize[0] = width;

        picture.linesize[1] = width >> 1; 

        picture.linesize[2] = width >> 1;

        break;

    case PIX_FMT_RGB24:

        size1 = (width * height * 3);

        if (size != size1)

            return -EIO;

        picture.data[0] = buf;

        picture.linesize[0] = width * 3;

        break;

    default:

        return -EIO;

    }

    

    if (get_frame_filename(filename, sizeof(filename), 

                           img->path, img->img_number) < 0)

        return -EIO;



    if (!img->is_pipe) {

        pb = &pb1;

        if (url_fopen(pb, filename, URL_WRONLY) < 0)

            return -EIO;

    } else {

        pb = &s->pb;

    }

    switch(img->img_fmt) {

    case IMGFMT_PGMYUV:

        ret = pgm_save(&picture, width, height, pb, 1);

        break;

    case IMGFMT_PGM:

        ret = pgm_save(&picture, width, height, pb, 0);

        break;

    case IMGFMT_YUV:

        ret = yuv_save(&picture, width, height, filename);

        break;

    case IMGFMT_PPM:

        ret = ppm_save(&picture, width, height, pb);

        break;

    }

    if (!img->is_pipe) {

        url_fclose(pb);

    }



    img->img_number++;

    return 0;

}
