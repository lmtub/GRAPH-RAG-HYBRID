static av_always_inline void rv40_strong_loop_filter(uint8_t *src,

                                                     const int step,

                                                     const int stride,

                                                     const int alpha,

                                                     const int lims,

                                                     const int dmode,

                                                     const int chroma)

{

    int i;



    for(i = 0; i < 4; i++, src += stride){

        int sflag, p0, q0, p1, q1;

        int t = src[0*step] - src[-1*step];



        if (!t)

            continue;



        sflag = (alpha * FFABS(t)) >> 7;

        if (sflag > 1)

            continue;



        p0 = (25*src[-3*step] + 26*src[-2*step] + 26*src[-1*step] +

              26*src[ 0*step] + 25*src[ 1*step] +

              rv40_dither_l[dmode + i]) >> 7;



        q0 = (25*src[-2*step] + 26*src[-1*step] + 26*src[ 0*step] +

              26*src[ 1*step] + 25*src[ 2*step] +

              rv40_dither_r[dmode + i]) >> 7;



        if (sflag) {

            p0 = av_clip(p0, src[-1*step] - lims, src[-1*step] + lims);

            q0 = av_clip(q0, src[ 0*step] - lims, src[ 0*step] + lims);

        }



        p1 = (25*src[-4*step] + 26*src[-3*step] + 26*src[-2*step] + 26*p0 +

              25*src[ 0*step] + rv40_dither_l[dmode + i]) >> 7;

        q1 = (25*src[-1*step] + 26*q0 + 26*src[ 1*step] + 26*src[ 2*step] +

              25*src[ 3*step] + rv40_dither_r[dmode + i]) >> 7;



        if (sflag) {

            p1 = av_clip(p1, src[-2*step] - lims, src[-2*step] + lims);

            q1 = av_clip(q1, src[ 1*step] - lims, src[ 1*step] + lims);

        }



        src[-2*step] = p1;

        src[-1*step] = p0;

        src[ 0*step] = q0;

        src[ 1*step] = q1;



        if(!chroma){

            src[-3*step] = (25*src[-1*step] + 26*src[-2*step] +

                            51*src[-3*step] + 26*src[-4*step] + 64) >> 7;

            src[ 2*step] = (25*src[ 0*step] + 26*src[ 1*step] +

                            51*src[ 2*step] + 26*src[ 3*step] + 64) >> 7;

        }

    }

}
