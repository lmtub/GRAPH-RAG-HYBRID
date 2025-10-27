int img_pad(AVPicture *dst, const AVPicture *src, int height, int width,

            int pix_fmt, int padtop, int padbottom, int padleft, int padright,

            int *color)

{

    uint8_t *optr, *iptr;

    int y_shift;

    int x_shift;

    int yheight;

    int i, y;



    if (pix_fmt < 0 || pix_fmt >= PIX_FMT_NB ||

        !is_yuv_planar(&pix_fmt_info[pix_fmt])) return -1;



    for (i = 0; i < 3; i++) {

        x_shift = i ? pix_fmt_info[pix_fmt].x_chroma_shift : 0;

        y_shift = i ? pix_fmt_info[pix_fmt].y_chroma_shift : 0;



        if (padtop || padleft) {

            memset(dst->data[i], color[i],

                dst->linesize[i] * (padtop >> y_shift) + (padleft >> x_shift));

        }



        if (padleft || padright || src) {

            if (src) { /* first line */

                iptr = src->data[i];

                optr = dst->data[i] + dst->linesize[i] * (padtop >> y_shift) +

                        (padleft >> x_shift);

                memcpy(optr, iptr, src->linesize[i]);

                iptr += src->linesize[i];

            }

            optr = dst->data[i] + dst->linesize[i] * (padtop >> y_shift) +

                (dst->linesize[i] - (padright >> x_shift));

            yheight = (height - 1 - (padtop + padbottom)) >> y_shift;

            for (y = 0; y < yheight; y++) {

                memset(optr, color[i], (padleft + padright) >> x_shift);

                if (src) {

                    memcpy(optr + ((padleft + padright) >> x_shift), iptr,

                        src->linesize[i]);

                    iptr += src->linesize[i];

                }

                optr += dst->linesize[i];

            }

        }



        if (padbottom || padright) {

            optr = dst->data[i] + dst->linesize[i] *

                ((height - padbottom) >> y_shift) - (padright >> x_shift);

            memset(optr, color[i],dst->linesize[i] *

                (padbottom >> y_shift) + (padright >> x_shift));

        }

    }

    return 0;

}
