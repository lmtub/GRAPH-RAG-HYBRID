enum AVPixelFormat choose_pixel_fmt(AVStream *st, AVCodec *codec, enum AVPixelFormat target)

{

    if (codec && codec->pix_fmts) {

        const enum AVPixelFormat *p = codec->pix_fmts;

        int has_alpha= av_pix_fmt_desc_get(target)->nb_components % 2 == 0;

        enum AVPixelFormat best= AV_PIX_FMT_NONE;

        if (st->codec->strict_std_compliance <= FF_COMPLIANCE_UNOFFICIAL) {

            if (st->codec->codec_id == AV_CODEC_ID_MJPEG) {

                p = (const enum AVPixelFormat[]) { AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_NONE };

            } else if (st->codec->codec_id == AV_CODEC_ID_LJPEG) {

                p = (const enum AVPixelFormat[]) { AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUV420P,

                                                 AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P, AV_PIX_FMT_BGRA, AV_PIX_FMT_NONE };

            }

        }

        for (; *p != AV_PIX_FMT_NONE; p++) {

            best= avcodec_find_best_pix_fmt_of_2(best, *p, target, has_alpha, NULL);

            if (*p == target)

                break;

        }

        if (*p == AV_PIX_FMT_NONE) {

            if (target != AV_PIX_FMT_NONE)

                av_log(NULL, AV_LOG_WARNING,

                       "Incompatible pixel format '%s' for codec '%s', auto-selecting format '%s'\n",

                       av_get_pix_fmt_name(target),

                       codec->name,

                       av_get_pix_fmt_name(best));

            return best;

        }

    }

    return target;

}
