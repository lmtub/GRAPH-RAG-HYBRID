static int yuv4_write_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVStream *st = s->streams[pkt->stream_index];

    AVIOContext *pb = s->pb;

    AVFrame *frame;

    int* first_pkt = s->priv_data;

    int width, height, h_chroma_shift, v_chroma_shift;

    int i;

    char buf2[Y4M_LINE_MAX + 1];

    uint8_t *ptr, *ptr1, *ptr2;



    frame = (AVFrame *)pkt->data;



    /* for the first packet we have to output the header as well */

    if (*first_pkt) {

        *first_pkt = 0;

        if (yuv4_generate_header(s, buf2) < 0) {

            av_log(s, AV_LOG_ERROR,

                   "Error. YUV4MPEG stream header write failed.\n");

            return AVERROR(EIO);

        } else {

            avio_write(pb, buf2, strlen(buf2));

        }

    }



    /* construct frame header */



    avio_printf(s->pb, "%s\n", Y4M_FRAME_MAGIC);



    width  = st->codecpar->width;

    height = st->codecpar->height;



    ptr = frame->data[0];



    switch (st->codecpar->format) {

    case AV_PIX_FMT_GRAY8:

    case AV_PIX_FMT_YUV411P:

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUV422P:

    case AV_PIX_FMT_YUV444P:

        break;

    case AV_PIX_FMT_GRAY16:

    case AV_PIX_FMT_YUV420P9:

    case AV_PIX_FMT_YUV422P9:

    case AV_PIX_FMT_YUV444P9:

    case AV_PIX_FMT_YUV420P10:

    case AV_PIX_FMT_YUV422P10:

    case AV_PIX_FMT_YUV444P10:

    case AV_PIX_FMT_YUV420P12:

    case AV_PIX_FMT_YUV422P12:

    case AV_PIX_FMT_YUV444P12:

    case AV_PIX_FMT_YUV420P14:

    case AV_PIX_FMT_YUV422P14:

    case AV_PIX_FMT_YUV444P14:

    case AV_PIX_FMT_YUV420P16:

    case AV_PIX_FMT_YUV422P16:

    case AV_PIX_FMT_YUV444P16:

        width *= 2;

        break;

    default:

        av_log(s, AV_LOG_ERROR, "The pixel format '%s' is not supported.\n",

               av_get_pix_fmt_name(st->codecpar->format));

        return AVERROR(EINVAL);

    }



    for (i = 0; i < height; i++) {

        avio_write(pb, ptr, width);

        ptr += frame->linesize[0];

    }



    if (st->codecpar->format != AV_PIX_FMT_GRAY8 &&

        st->codecpar->format != AV_PIX_FMT_GRAY16) {

        // Adjust for smaller Cb and Cr planes

        av_pix_fmt_get_chroma_sub_sample(st->codecpar->format, &h_chroma_shift,

                                         &v_chroma_shift);

        // Shift right, rounding up

        width  = AV_CEIL_RSHIFT(width,  h_chroma_shift);

        height = AV_CEIL_RSHIFT(height, v_chroma_shift);



        ptr1 = frame->data[1];

        ptr2 = frame->data[2];

        for (i = 0; i < height; i++) {     /* Cb */

            avio_write(pb, ptr1, width);

            ptr1 += frame->linesize[1];

        }

        for (i = 0; i < height; i++) {     /* Cr */

            avio_write(pb, ptr2, width);

            ptr2 += frame->linesize[2];

        }

    }



    return 0;

}
