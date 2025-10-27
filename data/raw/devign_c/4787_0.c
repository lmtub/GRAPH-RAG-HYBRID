static int set_format(void *obj, const char *name, int fmt, int search_flags,

                      enum AVOptionType type, const char *desc, int nb_fmts)

{

    void *target_obj;

    const AVOption *o = av_opt_find2(obj, name, NULL, 0,

                                     search_flags, &target_obj);

    int min, max;

    const AVClass *class = *(AVClass **)obj;



    if (!o || !target_obj)

        return AVERROR_OPTION_NOT_FOUND;

    if (o->type != type) {

        av_log(obj, AV_LOG_ERROR,

               "The value set by option '%s' is not a %s format", name, desc);

        return AVERROR(EINVAL);

    }



#if LIBAVUTIL_VERSION_MAJOR < 54

    if (class->version && class->version < AV_VERSION_INT(52, 11, 100)) {

        min = -1;

        max = nb_fmts-1;

    } else

#endif

    {

        min = FFMIN(o->min, -1);

        max = FFMAX(o->max, nb_fmts-1);

    }

    if (fmt < min || fmt > max) {

        av_log(obj, AV_LOG_ERROR,

               "Value %d for parameter '%s' out of %s format range [%d - %d]\n",

               fmt, name, desc, min, max);

        return AVERROR(ERANGE);

    }

    *(int *)(((uint8_t *)target_obj) + o->offset) = fmt;

    return 0;

}
