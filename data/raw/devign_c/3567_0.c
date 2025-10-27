static void filter_mb_mbaff_edgev( H264Context *h, uint8_t *pix, int stride, int16_t bS[4], int bsi, int qp ) {

    int i;

    int index_a = qp + h->slice_alpha_c0_offset;

    int alpha = (alpha_table+52)[index_a];

    int beta  = (beta_table+52)[qp + h->slice_beta_offset];

    for( i = 0; i < 8; i++, pix += stride) {

        const int bS_index = (i >> 1) * bsi;



        if( bS[bS_index] == 0 ) {

            continue;

        }



        if( bS[bS_index] < 4 ) {

            const int tc0 = (tc0_table+52)[index_a][bS[bS_index]];

            const int p0 = pix[-1];

            const int p1 = pix[-2];

            const int p2 = pix[-3];

            const int q0 = pix[0];

            const int q1 = pix[1];

            const int q2 = pix[2];



            if( FFABS( p0 - q0 ) < alpha &&

                FFABS( p1 - p0 ) < beta &&

                FFABS( q1 - q0 ) < beta ) {

                int tc = tc0;

                int i_delta;



                if( FFABS( p2 - p0 ) < beta ) {

                    if(tc0)

                    pix[-2] = p1 + av_clip( ( p2 + ( ( p0 + q0 + 1 ) >> 1 ) - ( p1 << 1 ) ) >> 1, -tc0, tc0 );

                    tc++;

                }

                if( FFABS( q2 - q0 ) < beta ) {

                    if(tc0)

                    pix[1] = q1 + av_clip( ( q2 + ( ( p0 + q0 + 1 ) >> 1 ) - ( q1 << 1 ) ) >> 1, -tc0, tc0 );

                    tc++;

                }



                i_delta = av_clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );

                pix[-1] = av_clip_uint8( p0 + i_delta );    /* p0' */

                pix[0]  = av_clip_uint8( q0 - i_delta );    /* q0' */

                tprintf(h->s.avctx, "filter_mb_mbaff_edgev i:%d, qp:%d, indexA:%d, alpha:%d, beta:%d, tc:%d\n# bS:%d -> [%02x, %02x, %02x, %02x, %02x, %02x] =>[%02x, %02x, %02x, %02x]\n", i, qp[qp_index], index_a, alpha, beta, tc, bS[bS_index], pix[-3], p1, p0, q0, q1, pix[2], p1, pix[-1], pix[0], q1);

            }

        }else{

            const int p0 = pix[-1];

            const int p1 = pix[-2];

            const int p2 = pix[-3];



            const int q0 = pix[0];

            const int q1 = pix[1];

            const int q2 = pix[2];



            if( FFABS( p0 - q0 ) < alpha &&

                FFABS( p1 - p0 ) < beta &&

                FFABS( q1 - q0 ) < beta ) {



                if(FFABS( p0 - q0 ) < (( alpha >> 2 ) + 2 )){

                    if( FFABS( p2 - p0 ) < beta)

                    {

                        const int p3 = pix[-4];

                        /* p0', p1', p2' */

                        pix[-1] = ( p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4 ) >> 3;

                        pix[-2] = ( p2 + p1 + p0 + q0 + 2 ) >> 2;

                        pix[-3] = ( 2*p3 + 3*p2 + p1 + p0 + q0 + 4 ) >> 3;

                    } else {

                        /* p0' */

                        pix[-1] = ( 2*p1 + p0 + q1 + 2 ) >> 2;

                    }

                    if( FFABS( q2 - q0 ) < beta)

                    {

                        const int q3 = pix[3];

                        /* q0', q1', q2' */

                        pix[0] = ( p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4 ) >> 3;

                        pix[1] = ( p0 + q0 + q1 + q2 + 2 ) >> 2;

                        pix[2] = ( 2*q3 + 3*q2 + q1 + q0 + p0 + 4 ) >> 3;

                    } else {

                        /* q0' */

                        pix[0] = ( 2*q1 + q0 + p1 + 2 ) >> 2;

                    }

                }else{

                    /* p0', q0' */

                    pix[-1] = ( 2*p1 + p0 + q1 + 2 ) >> 2;

                    pix[ 0] = ( 2*q1 + q0 + p1 + 2 ) >> 2;

                }

                tprintf(h->s.avctx, "filter_mb_mbaff_edgev i:%d, qp:%d, indexA:%d, alpha:%d, beta:%d\n# bS:4 -> [%02x, %02x, %02x, %02x, %02x, %02x] =>[%02x, %02x, %02x, %02x, %02x, %02x]\n", i, qp[qp_index], index_a, alpha, beta, p2, p1, p0, q0, q1, q2, pix[-3], pix[-2], pix[-1], pix[0], pix[1], pix[2]);

            }

        }

    }

}
