static opj_image_t *mj2_create_image(AVCodecContext *avctx, opj_cparameters_t *parameters)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avctx->pix_fmt);
    opj_image_cmptparm_t cmptparm[4] = {{0}};
    opj_image_t *img;
    int i;
    int sub_dx[4];
    int sub_dy[4];
    int numcomps;
    OPJ_COLOR_SPACE color_space = CLRSPC_UNKNOWN;
    sub_dx[0] = sub_dx[3] = 1;
    sub_dy[0] = sub_dy[3] = 1;
    sub_dx[1] = sub_dx[2] = 1 << desc->log2_chroma_w;
    sub_dy[1] = sub_dy[2] = 1 << desc->log2_chroma_h;
    numcomps = desc->nb_components;
    switch (avctx->pix_fmt) {
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_YA8:
    case AV_PIX_FMT_GRAY16:
    case AV_PIX_FMT_YA16:
        color_space = CLRSPC_GRAY;
        break;
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_RGB48:
    case AV_PIX_FMT_RGBA64:
    case AV_PIX_FMT_GBR24P:
    case AV_PIX_FMT_GBRP9:
    case AV_PIX_FMT_GBRP10:
    case AV_PIX_FMT_GBRP12:
    case AV_PIX_FMT_GBRP14:
    case AV_PIX_FMT_GBRP16:
    case AV_PIX_FMT_XYZ12:
        color_space = CLRSPC_SRGB;
        break;
    case AV_PIX_FMT_YUV410P:
    case AV_PIX_FMT_YUV411P:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV440P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVA420P:
    case AV_PIX_FMT_YUVA422P:
    case AV_PIX_FMT_YUVA444P:
    case AV_PIX_FMT_YUV420P9:
    case AV_PIX_FMT_YUV422P9:
    case AV_PIX_FMT_YUV444P9:
    case AV_PIX_FMT_YUVA420P9:
    case AV_PIX_FMT_YUVA422P9:
    case AV_PIX_FMT_YUVA444P9:
    case AV_PIX_FMT_YUV420P10:
    case AV_PIX_FMT_YUV422P10:
    case AV_PIX_FMT_YUV444P10:
    case AV_PIX_FMT_YUVA420P10:
    case AV_PIX_FMT_YUVA422P10:
    case AV_PIX_FMT_YUVA444P10:
    case AV_PIX_FMT_YUV420P12:
    case AV_PIX_FMT_YUV422P12:
    case AV_PIX_FMT_YUV444P12:
    case AV_PIX_FMT_YUV420P14:
    case AV_PIX_FMT_YUV422P14:
    case AV_PIX_FMT_YUV444P14:
    case AV_PIX_FMT_YUV420P16:
    case AV_PIX_FMT_YUV422P16:
    case AV_PIX_FMT_YUV444P16:
    case AV_PIX_FMT_YUVA420P16:
    case AV_PIX_FMT_YUVA422P16:
    case AV_PIX_FMT_YUVA444P16:
        color_space = CLRSPC_SYCC;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR,
               "The requested pixel format '%s' is not supported\n",
               av_get_pix_fmt_name(avctx->pix_fmt));
    }
    for (i = 0; i < numcomps; i++) {
        cmptparm[i].prec = desc->comp[i].depth_minus1 + 1;
        cmptparm[i].bpp  = desc->comp[i].depth_minus1 + 1;
        cmptparm[i].sgnd = 0;
        cmptparm[i].dx = sub_dx[i];
        cmptparm[i].dy = sub_dy[i];
        cmptparm[i].w = (avctx->width + sub_dx[i] - 1) / sub_dx[i];
        cmptparm[i].h = (avctx->height + sub_dy[i] - 1) / sub_dy[i];
    }
    img = opj_image_create(numcomps, cmptparm, color_space);
    // x0, y0 is the top left corner of the image
    // x1, y1 is the width, height of the reference grid
    img->x0 = 0;
    img->y0 = 0;
    img->x1 = (avctx->width  - 1) * parameters->subsampling_dx + 1;
    img->y1 = (avctx->height - 1) * parameters->subsampling_dy + 1;
    return img;
}