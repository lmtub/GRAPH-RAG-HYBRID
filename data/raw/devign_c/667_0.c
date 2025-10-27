static enum AVPixelFormat get_pixel_format(H264Context *h)

{

#define HWACCEL_MAX (CONFIG_H264_DXVA2_HWACCEL + \

                     CONFIG_H264_D3D11VA_HWACCEL + \

                     CONFIG_H264_VAAPI_HWACCEL + \

                     (CONFIG_H264_VDA_HWACCEL * 2) + \

                     CONFIG_H264_VDPAU_HWACCEL)

    enum AVPixelFormat pix_fmts[HWACCEL_MAX + 2], *fmt = pix_fmts;

    const enum AVPixelFormat *choices = pix_fmts;



    switch (h->sps.bit_depth_luma) {

    case 9:

        if (CHROMA444(h)) {

            if (h->avctx->colorspace == AVCOL_SPC_RGB) {

                *fmt++ = AV_PIX_FMT_GBRP9;

            } else

                *fmt++ = AV_PIX_FMT_YUV444P9;

        } else if (CHROMA422(h))

            *fmt++ = AV_PIX_FMT_YUV422P9;

        else

            *fmt++ = AV_PIX_FMT_YUV420P9;

        break;

    case 10:

        if (CHROMA444(h)) {

            if (h->avctx->colorspace == AVCOL_SPC_RGB) {

                *fmt++ = AV_PIX_FMT_GBRP10;

            } else

                *fmt++ = AV_PIX_FMT_YUV444P10;

        } else if (CHROMA422(h))

            *fmt++ = AV_PIX_FMT_YUV422P10;

        else

            *fmt++ = AV_PIX_FMT_YUV420P10;

        break;

    case 8:

#if CONFIG_H264_VDPAU_HWACCEL

        *fmt++ = AV_PIX_FMT_VDPAU;

#endif

        if (CHROMA444(h)) {

            if (h->avctx->colorspace == AVCOL_SPC_RGB)

                *fmt++ = AV_PIX_FMT_GBRP;

            else if (h->avctx->color_range == AVCOL_RANGE_JPEG)

                *fmt++ = AV_PIX_FMT_YUVJ444P;

            else

                *fmt++ = AV_PIX_FMT_YUV444P;

        } else if (CHROMA422(h)) {

            if (h->avctx->color_range == AVCOL_RANGE_JPEG)

                *fmt++ = AV_PIX_FMT_YUVJ422P;

            else

                *fmt++ = AV_PIX_FMT_YUV422P;

        } else {

#if CONFIG_H264_DXVA2_HWACCEL

            *fmt++ = AV_PIX_FMT_DXVA2_VLD;

#endif

#if CONFIG_H264_D3D11VA_HWACCEL

            *fmt++ = AV_PIX_FMT_D3D11VA_VLD;

#endif

#if CONFIG_H264_VAAPI_HWACCEL

            *fmt++ = AV_PIX_FMT_VAAPI;

#endif

#if CONFIG_H264_VDA_HWACCEL

            *fmt++ = AV_PIX_FMT_VDA_VLD;

            *fmt++ = AV_PIX_FMT_VDA;

#endif

            if (h->avctx->codec->pix_fmts)

                choices = h->avctx->codec->pix_fmts;

            else if (h->avctx->color_range == AVCOL_RANGE_JPEG)

                *fmt++ = AV_PIX_FMT_YUVJ420P;

            else

                *fmt++ = AV_PIX_FMT_YUV420P;

        }

        break;

    default:

        av_log(h->avctx, AV_LOG_ERROR,

               "Unsupported bit depth %d\n", h->sps.bit_depth_luma);

        return AVERROR_INVALIDDATA;

    }



    *fmt = AV_PIX_FMT_NONE;



    return ff_get_format(h->avctx, choices);

}
