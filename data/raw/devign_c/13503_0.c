int av_opt_set_pixel_fmt(void *obj, const char *name, enum AVPixelFormat fmt, int search_flags)

{

    return set_format(obj, name, fmt, search_flags, AV_OPT_TYPE_PIXEL_FMT, "pixel", AV_PIX_FMT_NB-1);

}
