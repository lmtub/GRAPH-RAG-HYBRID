static av_always_inline av_flatten void FUNCC(h264_loop_filter_luma)(uint8_t *p_pix, int xstride, int ystride, int inner_iters, int alpha, int beta, int8_t *tc0)

{

    pixel *pix = (pixel*)p_pix;

    int i, d;

    xstride >>= sizeof(pixel)-1;

    ystride >>= sizeof(pixel)-1;

    alpha <<= BIT_DEPTH - 8;

    beta  <<= BIT_DEPTH - 8;

    for( i = 0; i < 4; i++ ) {

        const int tc_orig = tc0[i] << (BIT_DEPTH - 8);

        if( tc_orig < 0 ) {

            pix += inner_iters*ystride;

            continue;

        }

        for( d = 0; d < inner_iters; d++ ) {

            const int p0 = pix[-1*xstride];

            const int p1 = pix[-2*xstride];

            const int p2 = pix[-3*xstride];

            const int q0 = pix[0];

            const int q1 = pix[1*xstride];

            const int q2 = pix[2*xstride];



            if( FFABS( p0 - q0 ) < alpha &&

                FFABS( p1 - p0 ) < beta &&

                FFABS( q1 - q0 ) < beta ) {



                int tc = tc_orig;

                int i_delta;



                if( FFABS( p2 - p0 ) < beta ) {

                    if(tc_orig)

                    pix[-2*xstride] = p1 + av_clip( (( p2 + ( ( p0 + q0 + 1 ) >> 1 ) ) >> 1) - p1, -tc_orig, tc_orig );

                    tc++;

                }

                if( FFABS( q2 - q0 ) < beta ) {

                    if(tc_orig)

                    pix[   xstride] = q1 + av_clip( (( q2 + ( ( p0 + q0 + 1 ) >> 1 ) ) >> 1) - q1, -tc_orig, tc_orig );

                    tc++;

                }



                i_delta = av_clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );

                pix[-xstride] = av_clip_pixel( p0 + i_delta );    /* p0' */

                pix[0]        = av_clip_pixel( q0 - i_delta );    /* q0' */

            }

            pix += ystride;

        }

    }

}
