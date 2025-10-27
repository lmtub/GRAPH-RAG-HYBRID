int avpicture_deinterlace(AVPicture *dst, const AVPicture *src,

                          enum AVPixelFormat pix_fmt, int width, int height)

{

    int i;



    if (pix_fmt != AV_PIX_FMT_YUV420P &&

        pix_fmt != AV_PIX_FMT_YUVJ420P &&

        pix_fmt != AV_PIX_FMT_YUV422P &&

        pix_fmt != AV_PIX_FMT_YUVJ422P &&

        pix_fmt != AV_PIX_FMT_YUV444P &&

        pix_fmt != AV_PIX_FMT_YUV411P &&

        pix_fmt != AV_PIX_FMT_GRAY8)

        return -1;

    if ((width & 3) != 0 || (height & 3) != 0)

        return -1;



    for(i=0;i<3;i++) {

        if (i == 1) {

            switch(pix_fmt) {

            case AV_PIX_FMT_YUVJ420P:

            case AV_PIX_FMT_YUV420P:

                width >>= 1;

                height >>= 1;

                break;

            case AV_PIX_FMT_YUV422P:

            case AV_PIX_FMT_YUVJ422P:

                width >>= 1;

                break;

            case AV_PIX_FMT_YUV411P:

                width >>= 2;

                break;

            default:

                break;

            }

            if (pix_fmt == AV_PIX_FMT_GRAY8) {

                break;

            }

        }

        if (src == dst) {

            deinterlace_bottom_field_inplace(dst->data[i], dst->linesize[i],

                                 width, height);

        } else {

            deinterlace_bottom_field(dst->data[i],dst->linesize[i],

                                        src->data[i], src->linesize[i],

                                        width, height);

        }

    }

    emms_c();

    return 0;

}
