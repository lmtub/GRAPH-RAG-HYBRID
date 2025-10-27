static inline void h264_loop_filter_luma_intra_c(uint8_t *pix, int xstride, int ystride, int alpha, int beta)

{

    int d;

    for( d = 0; d < 16; d++ ) {

        const int p2 = pix[-3*xstride];

        const int p1 = pix[-2*xstride];

        const int p0 = pix[-1*xstride];



        const int q0 = pix[ 0*xstride];

        const int q1 = pix[ 1*xstride];

        const int q2 = pix[ 2*xstride];



        if( FFABS( p0 - q0 ) < alpha &&

            FFABS( p1 - p0 ) < beta &&

            FFABS( q1 - q0 ) < beta ) {



            if(FFABS( p0 - q0 ) < (( alpha >> 2 ) + 2 )){

                if( FFABS( p2 - p0 ) < beta)

                {

                    const int p3 = pix[-4*xstride];

                    /* p0', p1', p2' */

                    pix[-1*xstride] = ( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3;

                    pix[-2*xstride] = ( p2 + p1 + p0 + q0 + 2 ) >> 2;

                    pix[-3*xstride] = ( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3;

                } else {

                    /* p0' */

                    pix[-1*xstride] = ( 2*p1 + p0 + q1 + 2 ) >> 2;

                }

                if( FFABS( q2 - q0 ) < beta)

                {

                    const int q3 = pix[3*xstride];

                    /* q0', q1', q2' */

                    pix[0*xstride] = ( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3;

                    pix[1*xstride] = ( p0 + q0 + q1 + q2 + 2 ) >> 2;

                    pix[2*xstride] = ( 2*q3 + 3*q2 + q1 + q0 + p0 + 4 ) >> 3;

                } else {

                    /* q0' */

                    pix[0*xstride] = ( 2*q1 + q0 + p1 + 2 ) >> 2;

                }

            }else{

                /* p0', q0' */

                pix[-1*xstride] = ( 2*p1 + p0 + q1 + 2 ) >> 2;

                pix[ 0*xstride] = ( 2*q1 + q0 + p1 + 2 ) >> 2;

            }

        }

        pix += ystride;

    }

}
