static int get_pix_fmt_score(enum AVPixelFormat dst_pix_fmt,

                              enum AVPixelFormat src_pix_fmt,

                              unsigned *lossp, unsigned consider)

{

    const AVPixFmtDescriptor *src_desc = av_pix_fmt_desc_get(src_pix_fmt);

    const AVPixFmtDescriptor *dst_desc = av_pix_fmt_desc_get(dst_pix_fmt);

    int src_color, dst_color;

    int src_min_depth, src_max_depth, dst_min_depth, dst_max_depth;

    int ret, loss, i, nb_components;

    int score = INT_MAX - 1;



    if (dst_pix_fmt >= AV_PIX_FMT_NB || dst_pix_fmt <= AV_PIX_FMT_NONE)

        return ~0;



    /* compute loss */

    *lossp = loss = 0;



    if (dst_pix_fmt == src_pix_fmt)

        return INT_MAX;



    if ((ret = get_pix_fmt_depth(&src_min_depth, &src_max_depth, src_pix_fmt)) < 0)

        return ret;

    if ((ret = get_pix_fmt_depth(&dst_min_depth, &dst_max_depth, dst_pix_fmt)) < 0)

        return ret;



    src_color = get_color_type(src_desc);

    dst_color = get_color_type(dst_desc);

    if (dst_pix_fmt == AV_PIX_FMT_PAL8)

        nb_components = FFMIN(src_desc->nb_components, 4);

    else

        nb_components = FFMIN(src_desc->nb_components, dst_desc->nb_components);



    for (i = 0; i < nb_components; i++) {

        int depth_minus1 = (dst_pix_fmt == AV_PIX_FMT_PAL8) ? 7/nb_components : (dst_desc->comp[i].depth - 1);

        if (src_desc->comp[i].depth - 1 > depth_minus1 && (consider & FF_LOSS_DEPTH)) {

            loss |= FF_LOSS_DEPTH;

            score -= 65536 >> depth_minus1;

        }

    }



    if (consider & FF_LOSS_RESOLUTION) {

        if (dst_desc->log2_chroma_w > src_desc->log2_chroma_w) {

            loss |= FF_LOSS_RESOLUTION;

            score -= 256 << dst_desc->log2_chroma_w;

        }

        if (dst_desc->log2_chroma_h > src_desc->log2_chroma_h) {

            loss |= FF_LOSS_RESOLUTION;

            score -= 256 << dst_desc->log2_chroma_h;

        }

        // don't favor 422 over 420 if downsampling is needed, because 420 has much better support on the decoder side

        if (dst_desc->log2_chroma_w == 1 && src_desc->log2_chroma_w == 0 &&

            dst_desc->log2_chroma_h == 1 && src_desc->log2_chroma_h == 0 ) {

            score += 512;

        }

    }



    if(consider & FF_LOSS_COLORSPACE)

    switch(dst_color) {

    case FF_COLOR_RGB:

        if (src_color != FF_COLOR_RGB &&

            src_color != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_GRAY:

        if (src_color != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_YUV:

        if (src_color != FF_COLOR_YUV)

            loss |= FF_LOSS_COLORSPACE;

        break;

    case FF_COLOR_YUV_JPEG:

        if (src_color != FF_COLOR_YUV_JPEG &&

            src_color != FF_COLOR_YUV &&

            src_color != FF_COLOR_GRAY)

            loss |= FF_LOSS_COLORSPACE;

        break;

    default:

        /* fail safe test */

        if (src_color != dst_color)

            loss |= FF_LOSS_COLORSPACE;

        break;

    }

    if(loss & FF_LOSS_COLORSPACE)

        score -= (nb_components * 65536) >> FFMIN(dst_desc->comp[0].depth - 1, src_desc->comp[0].depth - 1);



    if (dst_color == FF_COLOR_GRAY &&

        src_color != FF_COLOR_GRAY && (consider & FF_LOSS_CHROMA)) {

        loss |= FF_LOSS_CHROMA;

        score -= 2 * 65536;

    }

    if (!pixdesc_has_alpha(dst_desc) && (pixdesc_has_alpha(src_desc) && (consider & FF_LOSS_ALPHA))) {

        loss |= FF_LOSS_ALPHA;

        score -= 65536;

    }

    if (dst_pix_fmt == AV_PIX_FMT_PAL8 && (consider & FF_LOSS_COLORQUANT) &&

        (src_pix_fmt != AV_PIX_FMT_PAL8 && (src_color != FF_COLOR_GRAY || (pixdesc_has_alpha(src_desc) && (consider & FF_LOSS_ALPHA))))) {

        loss |= FF_LOSS_COLORQUANT;

        score -= 65536;

    }



    *lossp = loss;

    return score;

}
