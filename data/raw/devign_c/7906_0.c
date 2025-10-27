int av_image_get_linesize(enum PixelFormat pix_fmt, int width, int plane)

{

    const AVPixFmtDescriptor *desc = &av_pix_fmt_descriptors[pix_fmt];

    int max_step     [4];       /* max pixel step for each plane */

    int max_step_comp[4];       /* the component for each plane which has the max pixel step */

    int s, linesize;



    if ((unsigned)pix_fmt >= PIX_FMT_NB || desc->flags & PIX_FMT_HWACCEL)

        return AVERROR(EINVAL);



    av_image_fill_max_pixsteps(max_step, max_step_comp, desc);

    s = (max_step_comp[plane] == 1 || max_step_comp[plane] == 2) ? desc->log2_chroma_w : 0;

    linesize = max_step[plane] * (((width + (1 << s) - 1)) >> s);

    if (desc->flags & PIX_FMT_BITSTREAM)

        linesize = (linesize + 7) >> 3;

    return linesize;

}
