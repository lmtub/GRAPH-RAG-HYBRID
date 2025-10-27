static int yuv4_write_header(AVFormatContext *s)

{

    int *first_pkt = s->priv_data;



    if (s->nb_streams != 1)

        return AVERROR(EIO);



    if (s->streams[0]->codecpar->codec_id != AV_CODEC_ID_WRAPPED_AVFRAME) {

        av_log(s, AV_LOG_ERROR, "ERROR: Codec not supported.\n");

        return AVERROR_INVALIDDATA;

    }



    switch (s->streams[0]->codecpar->format) {

    case AV_PIX_FMT_YUV411P:

        av_log(s, AV_LOG_WARNING, "Warning: generating rarely used 4:1:1 YUV "

               "stream, some mjpegtools might not work.\n");

        break;

    case AV_PIX_FMT_GRAY8:

    case AV_PIX_FMT_GRAY16:

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUV422P:

    case AV_PIX_FMT_YUV444P:

        break;

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

        if (s->strict_std_compliance >= FF_COMPLIANCE_NORMAL) {

            av_log(s, AV_LOG_ERROR, "'%s' is not an official yuv4mpegpipe pixel format. "

                   "Use '-strict -1' to encode to this pixel format.\n",

                   av_get_pix_fmt_name(s->streams[0]->codecpar->format));

            return AVERROR(EINVAL);

        }

        av_log(s, AV_LOG_WARNING, "Warning: generating non standard YUV stream. "

               "Mjpegtools will not work.\n");

        break;

    default:

        av_log(s, AV_LOG_ERROR, "ERROR: yuv4mpeg can only handle "

               "yuv444p, yuv422p, yuv420p, yuv411p and gray8 pixel formats. "

               "And using 'strict -1' also yuv444p9, yuv422p9, yuv420p9, "

               "yuv444p10, yuv422p10, yuv420p10, "

               "yuv444p12, yuv422p12, yuv420p12, "

               "yuv444p14, yuv422p14, yuv420p14, "

               "yuv444p16, yuv422p16, yuv420p16 "

               "and gray16 pixel formats. "

               "Use -pix_fmt to select one.\n");

        return AVERROR(EIO);

    }



    *first_pkt = 1;

    return 0;

}
