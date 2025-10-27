int av_opt_set(void *obj, const char *name, const char *val, int search_flags)

{

    int ret;

    void *dst, *target_obj;

    const AVOption *o = av_opt_find2(obj, name, NULL, 0, search_flags, &target_obj);

    if (!o || !target_obj)

        return AVERROR_OPTION_NOT_FOUND;

    if (!val && (o->type != AV_OPT_TYPE_STRING &&

                 o->type != AV_OPT_TYPE_PIXEL_FMT && o->type != AV_OPT_TYPE_SAMPLE_FMT &&

                 o->type != AV_OPT_TYPE_IMAGE_SIZE && o->type != AV_OPT_TYPE_VIDEO_RATE &&

                 o->type != AV_OPT_TYPE_DURATION && o->type != AV_OPT_TYPE_COLOR &&

                 o->type != AV_OPT_TYPE_CHANNEL_LAYOUT))

        return AVERROR(EINVAL);



    dst = ((uint8_t*)target_obj) + o->offset;

    switch (o->type) {

    case AV_OPT_TYPE_STRING:   return set_string(obj, o, val, dst);

    case AV_OPT_TYPE_BINARY:   return set_string_binary(obj, o, val, dst);

    case AV_OPT_TYPE_FLAGS:

    case AV_OPT_TYPE_INT:

    case AV_OPT_TYPE_INT64:

    case AV_OPT_TYPE_FLOAT:

    case AV_OPT_TYPE_DOUBLE:

    case AV_OPT_TYPE_RATIONAL: return set_string_number(obj, target_obj, o, val, dst);

    case AV_OPT_TYPE_IMAGE_SIZE:

        if (!val || !strcmp(val, "none")) {

            *(int *)dst = *((int *)dst + 1) = 0;

            return 0;

        }

        ret = av_parse_video_size(dst, ((int *)dst) + 1, val);

        if (ret < 0)

            av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as image size\n", val);

        return ret;

    case AV_OPT_TYPE_VIDEO_RATE:

        if (!val) {

            ret = AVERROR(EINVAL);

        } else {

            ret = av_parse_video_rate(dst, val);

        }

        if (ret < 0)

            av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as video rate\n", val);

        return ret;

    case AV_OPT_TYPE_PIXEL_FMT:

        if (!val || !strcmp(val, "none")) {

            ret = AV_PIX_FMT_NONE;

        } else {

            ret = av_get_pix_fmt(val);

            if (ret == AV_PIX_FMT_NONE) {

                char *tail;

                ret = strtol(val, &tail, 0);

                if (*tail || (unsigned)ret >= AV_PIX_FMT_NB) {

                    av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as pixel format\n", val);

                    return AVERROR(EINVAL);

                }

            }

        }

        *(enum AVPixelFormat *)dst = ret;

        return 0;

    case AV_OPT_TYPE_SAMPLE_FMT:

        if (!val || !strcmp(val, "none")) {

            ret = AV_SAMPLE_FMT_NONE;

        } else {

            ret = av_get_sample_fmt(val);

            if (ret == AV_SAMPLE_FMT_NONE) {

                char *tail;

                ret = strtol(val, &tail, 0);

                if (*tail || (unsigned)ret >= AV_SAMPLE_FMT_NB) {

                    av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as sample format\n", val);

                    return AVERROR(EINVAL);

                }

            }

        }

        *(enum AVSampleFormat *)dst = ret;

        return 0;

    case AV_OPT_TYPE_DURATION:

        if (!val) {

            *(int64_t *)dst = 0;

            return 0;

        } else {

            if ((ret = av_parse_time(dst, val, 1)) < 0)

                av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as duration\n", val);

            return ret;

        }

        break;

    case AV_OPT_TYPE_COLOR:

        if (!val) {

            return 0;

        } else {

            ret = av_parse_color(dst, val, -1, obj);

            if (ret < 0)

                av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as color\n", val);

            return ret;

        }

        break;

    case AV_OPT_TYPE_CHANNEL_LAYOUT:

        if (!val || !strcmp(val, "none")) {

            *(int64_t *)dst = 0;

        } else {

#if FF_API_GET_CHANNEL_LAYOUT_COMPAT

            int64_t cl = ff_get_channel_layout(val, 0);

#else

            int64_t cl = av_get_channel_layout(val);

#endif

            if (!cl) {

                av_log(obj, AV_LOG_ERROR, "Unable to parse option value \"%s\" as channel layout\n", val);

                ret = AVERROR(EINVAL);

            }

            *(int64_t *)dst = cl;

            return ret;

        }

        break;

    }



    av_log(obj, AV_LOG_ERROR, "Invalid option type.\n");

    return AVERROR(EINVAL);

}
