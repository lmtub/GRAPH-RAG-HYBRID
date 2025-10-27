const char *avcodec_get_pix_fmt_name(enum PixelFormat pix_fmt)

{

    if (pix_fmt < 0 || pix_fmt >= PIX_FMT_NB)

        return NULL;

    else

        return av_pix_fmt_descriptors[pix_fmt].name;

}
