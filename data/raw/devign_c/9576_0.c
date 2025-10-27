AVFilterFormats *avfilter_all_colorspaces(void)

{

    return avfilter_make_format_list(35,

                PIX_FMT_YUV444P,  PIX_FMT_YUV422P,  PIX_FMT_YUV420P,

                PIX_FMT_YUV411P,  PIX_FMT_YUV410P,

                PIX_FMT_YUYV422,  PIX_FMT_UYVY422,  PIX_FMT_UYYVYY411,

                PIX_FMT_YUVJ444P, PIX_FMT_YUVJ422P, PIX_FMT_YUVJ420P,

                PIX_FMT_YUV440P,  PIX_FMT_YUVJ440P,

                PIX_FMT_RGB32,    PIX_FMT_BGR32,

                PIX_FMT_RGB32_1,  PIX_FMT_BGR32_1,

                PIX_FMT_RGB24,    PIX_FMT_BGR24,

                PIX_FMT_RGB565,   PIX_FMT_BGR565,

                PIX_FMT_RGB555,   PIX_FMT_BGR555,

                PIX_FMT_RGB8,     PIX_FMT_BGR8,

                PIX_FMT_RGB4_BYTE,PIX_FMT_BGR4_BYTE,

                PIX_FMT_GRAY16BE, PIX_FMT_GRAY16LE,

                PIX_FMT_GRAY8,    PIX_FMT_PAL8,

                PIX_FMT_MONOWHITE,PIX_FMT_MONOBLACK

                PIX_FMT_NV12,     PIX_FMT_NV21);

}
