static void filter_mb_edgeh( H264Context *h, uint8_t *pix, int stride, int bS[4], int qp ) {

    int i, d;

    const int index_a = clip( qp + h->slice_alpha_c0_offset, 0, 51 );

    const int alpha = alpha_table[index_a];

    const int beta  = beta_table[clip( qp + h->slice_beta_offset, 0, 51 )];

    const int pix_next  = stride;



    for( i = 0; i < 4; i++ ) {

        if( bS[i] == 0 ) {

            pix += 4;

            continue;

        }



        /* 4px edge length */

        for( d = 0; d < 4; d++ ) {

            const uint8_t p0 = pix[-1*pix_next];

            const uint8_t p1 = pix[-2*pix_next];

            const uint8_t p2 = pix[-3*pix_next];

            const uint8_t q0 = pix[0];

            const uint8_t q1 = pix[1*pix_next];

            const uint8_t q2 = pix[2*pix_next];



            if( abs( p0 - q0 ) >= alpha ||

                abs( p1 - p0 ) >= beta ||

                abs( q1 - q0 ) >= beta ) {

                pix++;

                continue;

            }



            if( bS[i] < 4 ) {

                const int tc0 = tc0_table[index_a][bS[i] - 1];

                int tc = tc0;

                int i_delta;



                if( abs( p2 - p0 ) < beta ) {

                    pix[-2*pix_next] = p1 + clip( ( p2 + ( ( p0 + q0 + 1 ) >> 1 ) - ( p1 << 1 ) ) >> 1, -tc0, tc0 );

                    tc++;

                }

                if( abs( q2 - q0 ) < beta ) {

                    pix[pix_next] = q1 + clip( ( q2 + ( ( p0 + q0 + 1 ) >> 1 ) - ( q1 << 1 ) ) >> 1, -tc0, tc0 );

                    tc++;

                }



                i_delta = clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );

                pix[-pix_next] = clip( p0 + i_delta, 0, 255 );    /* p0' */

                pix[0]         = clip( q0 - i_delta, 0, 255 );    /* q0' */

            }

            else

            {

                const uint8_t p3 = pix[-4*pix_next];

                const uint8_t q3 = pix[ 3*pix_next];

                const int c  = abs( p0 - q0 ) < (( alpha >> 2 ) + 2 );



                if( abs( p2 - p0 ) < beta && c ) {

                    /* p0', p1', p2' */

                    pix[-1*pix_next] = ( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3;

                    pix[-2*pix_next] = ( p2 + p1 + p0 + q0 + 2 ) >> 2;

                    pix[-3*pix_next] = ( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3;

                } else {

                    /* p0' */

                    pix[-1*pix_next] = ( 2*p1 + p0 + q1 + 2 ) >> 2;

                }

                if( abs( q2 - q0 ) < beta && c ) {

                    /* q0', q1', q2' */

                    pix[0*pix_next] = ( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3;

                    pix[1*pix_next] = ( p0 + q0 + q1 + q2 + 2 ) >> 2;

                    pix[2*pix_next] = ( 2*q3 + 3*q2 + q1 + q0 + p0 + 4 ) >> 3;

                } else {

                    /* q0' */

                    pix[0*pix_next] = ( 2*q1 + q0 + p1 + 2 ) >> 2;

                }

            }

            pix++;

        }

    }

}
