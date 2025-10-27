int av_image_get_linesize(enum AVPixelFormat pix_fmt, int width, int plane)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);

    int max_step     [4];       /* max pixel step for each plane */

    int max_step_comp[4];       /* the component for each plane which has the max pixel step */



    if ((unsigned)pix_fmt >= AV_PIX_FMT_NB || desc->flags & AV_PIX_FMT_FLAG_HWACCEL)

        return AVERROR(EINVAL);



    av_image_fill_max_pixsteps(max_step, max_step_comp, desc);

    return image_get_linesize(width, plane, max_step[plane], max_step_comp[plane], desc);

}
