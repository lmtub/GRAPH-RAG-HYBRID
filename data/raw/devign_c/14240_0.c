static int set_format(void *obj, const char *name, int fmt, int search_flags,

                      enum AVOptionType type, const char *desc, int max)

{

    void *target_obj;

    const AVOption *o = av_opt_find2(obj, name, NULL, 0,

                                     search_flags, &target_obj);

    if (!o || !target_obj)

        return AVERROR_OPTION_NOT_FOUND;

    if (o->type != type) {

        av_log(obj, AV_LOG_ERROR,

               "The value set by option '%s' is not a %s format", name, desc);

        return AVERROR(EINVAL);

    }



    if (fmt < -1 || fmt > max) {

        av_log(obj, AV_LOG_ERROR,

               "Value %d for parameter '%s' out of %s format range [-1 - %d]\n",

               fmt, name, desc, max);

        return AVERROR(ERANGE);

    }

    *(int *)(((uint8_t *)target_obj) + o->offset) = fmt;

    return 0;

}
