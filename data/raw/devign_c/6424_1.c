static void write_picture(AVFormatContext *s, int index, AVPicture *picture, 

                          int pix_fmt, int w, int h)

{

    UINT8 *buf, *src, *dest;

    int size, j, i;



    size = avpicture_get_size(pix_fmt, w, h);

    buf = malloc(size);

    if (!buf)

        return;



    /* XXX: not efficient, should add test if we can take

       directly the AVPicture */

    switch(pix_fmt) {

    case PIX_FMT_YUV420P:

        dest = buf;

        for(i=0;i<3;i++) {

            if (i == 1) {

                w >>= 1;

                h >>= 1;

            }

            src = picture->data[i];

            for(j=0;j<h;j++) {

                memcpy(dest, src, w);

                dest += w;

                src += picture->linesize[i];

            }

        }

        break;

    case PIX_FMT_YUV422P:

        size = (w * h) * 2; 

        buf = malloc(size);

        dest = buf;

        for(i=0;i<3;i++) {

            if (i == 1) {

                w >>= 1;

            }

            src = picture->data[i];

            for(j=0;j<h;j++) {

                memcpy(dest, src, w);

                dest += w;

                src += picture->linesize[i];

            }

        }

        break;

    case PIX_FMT_YUV444P:

        size = (w * h) * 3; 

        buf = malloc(size);

        dest = buf;

        for(i=0;i<3;i++) {

            src = picture->data[i];

            for(j=0;j<h;j++) {

                memcpy(dest, src, w);

                dest += w;

                src += picture->linesize[i];

            }

        }

        break;

    case PIX_FMT_YUV422:

        size = (w * h) * 2; 

        buf = malloc(size);

        dest = buf;

        src = picture->data[0];

        for(j=0;j<h;j++) {

            memcpy(dest, src, w * 2);

            dest += w * 2;

            src += picture->linesize[0];

        }

        break;

    case PIX_FMT_RGB24:

    case PIX_FMT_BGR24:

        size = (w * h) * 3; 

        buf = malloc(size);

        dest = buf;

        src = picture->data[0];

        for(j=0;j<h;j++) {

            memcpy(dest, src, w * 3);

            dest += w * 3;

            src += picture->linesize[0];

        }

        break;

    default:

        return;

    }

    s->format->write_packet(s, index, buf, size);

    free(buf);

}
