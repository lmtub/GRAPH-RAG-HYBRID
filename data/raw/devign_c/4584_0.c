int avcodec_get_pix_fmt_loss(enum PixelFormat dst_pix_fmt, enum PixelFormat src_pix_fmt,

                             int has_alpha)

{

    const PixFmtInfo *pf, *ps;

    const AVPixFmtDescriptor *src_desc = &av_pix_fmt_descriptors[src_pix_fmt];

    const AVPixFmtDescriptor *dst_desc = &av_pix_fmt_descriptors[dst_pix_fmt];

    int loss;



    ps = &pix_fmt_info[src_pix_fmt];



    /* compute loss */

    loss = 0;

    pf = &pix_fmt_info[dst_pix_fmt];

    if (pf->depth < ps->depth ||

        ((dst_pix_fmt == PIX_FMT_RGB555BE || dst_pix_fmt == PIX_FMT_RGB555LE ||

          dst_pix_fmt == PIX_FMT_BGR555BE || dst_pix_fmt == PIX_FMT_BGR555LE) &&

         (src_pix_fmt == PIX_FMT_RGB565BE || src_pix_fmt == PIX_FMT_RGB565LE ||

          src_pix_fmt == PIX_FMT_BGR565BE || src_pix_fmt == PIX_FMT_BGR565LE)))

        loss |= FF_LOSS_DEPTH;

    if (dst_desc->log2_chroma_w > src_desc->log2_chroma_w ||

        dst_desc->log2_chroma_h > src_desc->log2_chroma_h)

        loss |= FF_LOSS_RESOLUTION;

    switch(pf->color_type) {

    case FF_COLOR_RGB:

        if (ps->color_type != FF_COLOR_RGB &&

            ps->color_type != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_GRAY:

        if (ps->color_type != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_YUV:

        if (ps->color_type != FF_COLOR_YUV)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_YUV_JPEG:

        if (ps->color_type != FF_COLOR_YUV_JPEG &&

            ps->color_type != FF_COLOR_YUV &&

            ps->color_type != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    default:

        /* fail safe test */

        if (ps->color_type != pf->color_type)

            loss |= FF_LOSS_COLORSPACE;

        break;

    }

    if (pf->color_type == FF_COLOR_GRAY &&

        ps->color_type != FF_COLOR_GRAY)

        loss |= FF_LOSS_CHROMA;

    if (!pf->is_alpha && (ps->is_alpha && has_alpha))

        loss |= FF_LOSS_ALPHA;

    if (pf->pixel_type == FF_PIXEL_PALETTE &&

        (ps->pixel_type != FF_PIXEL_PALETTE && ps->color_type != FF_COLOR_GRAY))

        loss |= FF_LOSS_COLORQUANT;

    return loss;

}
