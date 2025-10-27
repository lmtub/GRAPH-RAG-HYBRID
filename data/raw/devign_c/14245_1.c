static int yuv4_read_header(AVFormatContext *s)

{

    char header[MAX_YUV4_HEADER + 10];  // Include headroom for

                                        // the longest option

    char *tokstart, *tokend, *header_end;

    int i;

    AVIOContext *pb = s->pb;

    int width = -1, height  = -1, raten   = 0,

        rated =  0, aspectn =  0, aspectd = 0;

    enum AVPixelFormat pix_fmt = AV_PIX_FMT_NONE, alt_pix_fmt = AV_PIX_FMT_NONE;

    enum AVChromaLocation chroma_sample_location = AVCHROMA_LOC_UNSPECIFIED;

    AVStream *st;

    enum AVFieldOrder field_order;



    for (i = 0; i < MAX_YUV4_HEADER; i++) {

        header[i] = avio_r8(pb);

        if (header[i] == '\n') {

            header[i + 1] = 0x20;  // Add a space after last option.

                                   // Makes parsing "444" vs "444alpha" easier.

            header[i + 2] = 0;

            break;

        }

    }

    if (i == MAX_YUV4_HEADER)

        return -1;

    if (strncmp(header, Y4M_MAGIC, strlen(Y4M_MAGIC)))

        return -1;



    header_end = &header[i + 1]; // Include space

    for (tokstart = &header[strlen(Y4M_MAGIC) + 1];

         tokstart < header_end; tokstart++) {

        if (*tokstart == 0x20)

            continue;

        switch (*tokstart++) {

        case 'W': // Width. Required.

            width    = strtol(tokstart, &tokend, 10);

            tokstart = tokend;

            break;

        case 'H': // Height. Required.

            height   = strtol(tokstart, &tokend, 10);

            tokstart = tokend;

            break;

        case 'C': // Color space

            if (strncmp("420jpeg", tokstart, 7) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P;

                chroma_sample_location = AVCHROMA_LOC_CENTER;

            } else if (strncmp("420mpeg2", tokstart, 8) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P;

                chroma_sample_location = AVCHROMA_LOC_LEFT;

            } else if (strncmp("420paldv", tokstart, 8) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P;

                chroma_sample_location = AVCHROMA_LOC_TOPLEFT;

            } else if (strncmp("420p16", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P16;

            } else if (strncmp("422p16", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV422P16;

            } else if (strncmp("444p16", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV444P16;

            } else if (strncmp("420p14", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P14;

            } else if (strncmp("422p14", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV422P14;

            } else if (strncmp("444p14", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV444P14;

            } else if (strncmp("420p12", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P12;

            } else if (strncmp("422p12", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV422P12;

            } else if (strncmp("444p12", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV444P12;

            } else if (strncmp("420p10", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P10;

            } else if (strncmp("422p10", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV422P10;

            } else if (strncmp("444p10", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_YUV444P10;

            } else if (strncmp("420p9", tokstart, 5) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P9;

            } else if (strncmp("422p9", tokstart, 5) == 0) {

                pix_fmt = AV_PIX_FMT_YUV422P9;

            } else if (strncmp("444p9", tokstart, 5) == 0) {

                pix_fmt = AV_PIX_FMT_YUV444P9;

            } else if (strncmp("420", tokstart, 3) == 0) {

                pix_fmt = AV_PIX_FMT_YUV420P;

                chroma_sample_location = AVCHROMA_LOC_CENTER;

            } else if (strncmp("411", tokstart, 3) == 0) {

                pix_fmt = AV_PIX_FMT_YUV411P;

            } else if (strncmp("422", tokstart, 3) == 0) {

                pix_fmt = AV_PIX_FMT_YUV422P;

            } else if (strncmp("444alpha", tokstart, 8) == 0 ) {

                av_log(s, AV_LOG_ERROR, "Cannot handle 4:4:4:4 "

                       "YUV4MPEG stream.\n");

                return -1;

            } else if (strncmp("444", tokstart, 3) == 0) {

                pix_fmt = AV_PIX_FMT_YUV444P;

            } else if (strncmp("mono16", tokstart, 6) == 0) {

                pix_fmt = AV_PIX_FMT_GRAY16;

            } else if (strncmp("mono", tokstart, 4) == 0) {

                pix_fmt = AV_PIX_FMT_GRAY8;

            } else {

                av_log(s, AV_LOG_ERROR, "YUV4MPEG stream contains an unknown "

                       "pixel format.\n");

                return -1;

            }

            while (tokstart < header_end && *tokstart != 0x20)

                tokstart++;

            break;

        case 'I': // Interlace type

            switch (*tokstart++){

            case '?':

                field_order = AV_FIELD_UNKNOWN;

                break;

            case 'p':

                field_order = AV_FIELD_PROGRESSIVE;

                break;

            case 't':

                field_order = AV_FIELD_TT;

                break;

            case 'b':

                field_order = AV_FIELD_BB;

                break;

            case 'm':

                av_log(s, AV_LOG_ERROR, "YUV4MPEG stream contains mixed "

                       "interlaced and non-interlaced frames.\n");

            default:

                av_log(s, AV_LOG_ERROR, "YUV4MPEG has invalid header.\n");

                return AVERROR(EINVAL);

            }

            break;

        case 'F': // Frame rate

            sscanf(tokstart, "%d:%d", &raten, &rated); // 0:0 if unknown

            while (tokstart < header_end && *tokstart != 0x20)

                tokstart++;

            break;

        case 'A': // Pixel aspect

            sscanf(tokstart, "%d:%d", &aspectn, &aspectd); // 0:0 if unknown

            while (tokstart < header_end && *tokstart != 0x20)

                tokstart++;

            break;

        case 'X': // Vendor extensions

            if (strncmp("YSCSS=", tokstart, 6) == 0) {

                // Older nonstandard pixel format representation

                tokstart += 6;

                if (strncmp("420JPEG", tokstart, 7) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P;

                else if (strncmp("420MPEG2", tokstart, 8) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P;

                else if (strncmp("420PALDV", tokstart, 8) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P;

                else if (strncmp("420P9", tokstart, 5) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P9;

                else if (strncmp("422P9", tokstart, 5) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV422P9;

                else if (strncmp("444P9", tokstart, 5) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV444P9;

                else if (strncmp("420P10", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P10;

                else if (strncmp("422P10", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV422P10;

                else if (strncmp("444P10", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV444P10;

                else if (strncmp("420P12", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P12;

                else if (strncmp("422P12", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV422P12;

                else if (strncmp("444P12", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV444P12;

                else if (strncmp("420P14", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P14;

                else if (strncmp("422P14", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV422P14;

                else if (strncmp("444P14", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV444P14;

                else if (strncmp("420P16", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV420P16;

                else if (strncmp("422P16", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV422P16;

                else if (strncmp("444P16", tokstart, 6) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV444P16;

                else if (strncmp("411", tokstart, 3) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV411P;

                else if (strncmp("422", tokstart, 3) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV422P;

                else if (strncmp("444", tokstart, 3) == 0)

                    alt_pix_fmt = AV_PIX_FMT_YUV444P;

            }

            while (tokstart < header_end && *tokstart != 0x20)

                tokstart++;

            break;

        }

    }



    if (width == -1 || height == -1) {

        av_log(s, AV_LOG_ERROR, "YUV4MPEG has invalid header.\n");

        return -1;

    }



    if (pix_fmt == AV_PIX_FMT_NONE) {

        if (alt_pix_fmt == AV_PIX_FMT_NONE)

            pix_fmt = AV_PIX_FMT_YUV420P;

        else

            pix_fmt = alt_pix_fmt;

    }



    if (raten <= 0 || rated <= 0) {

        // Frame rate unknown

        raten = 25;

        rated = 1;

    }



    if (aspectn == 0 && aspectd == 0) {

        // Pixel aspect unknown

        aspectd = 1;

    }



    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    st->codec->width  = width;

    st->codec->height = height;

    av_reduce(&raten, &rated, raten, rated, (1UL << 31) - 1);

    avpriv_set_pts_info(st, 64, rated, raten);

    st->avg_frame_rate                = av_inv_q(st->time_base);

    st->codec->pix_fmt                = pix_fmt;

    st->codec->codec_type             = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id               = AV_CODEC_ID_RAWVIDEO;

    st->sample_aspect_ratio           = (AVRational){ aspectn, aspectd };

    st->codec->chroma_sample_location = chroma_sample_location;

    st->codec->field_order            = field_order;



    return 0;

}
