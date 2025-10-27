int avpicture_layout(const AVPicture* src, enum AVPixelFormat pix_fmt,

                     int width, int height,

                     unsigned char *dest, int dest_size)

{

    int i, j, nb_planes = 0, linesizes[4];

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(pix_fmt);

    int size = avpicture_get_size(pix_fmt, width, height);



    if (size > dest_size || size < 0)

        return AVERROR(EINVAL);



    for (i = 0; i < desc->nb_components; i++)

        nb_planes = FFMAX(desc->comp[i].plane, nb_planes);



    nb_planes++;



    av_image_fill_linesizes(linesizes, pix_fmt, width);

    for (i = 0; i < nb_planes; i++) {

        int h, shift = (i == 1 || i == 2) ? desc->log2_chroma_h : 0;

        const unsigned char *s = src->data[i];

        h = (height + (1 << shift) - 1) >> shift;



        for (j = 0; j < h; j++) {

            memcpy(dest, s, linesizes[i]);

            dest += linesizes[i];

            s += src->linesize[i];

        }

    }



    if (desc->flags & AV_PIX_FMT_FLAG_PAL)

        memcpy((unsigned char *)(((size_t)dest + 3) & ~3),

               src->data[1], 256 * 4);



    return size;

}
