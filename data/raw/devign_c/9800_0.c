static void filter_mb_mbaff_edgecv( H264Context *h, uint8_t *pix, int stride, int16_t bS[4], int bsi, int qp ) {

    int i;

    int index_a = qp + h->slice_alpha_c0_offset;

    int alpha = (alpha_table+52)[index_a];

    int beta  = (beta_table+52)[qp + h->slice_beta_offset];

    for( i = 0; i < 4; i++, pix += stride) {

        const int bS_index = i*bsi;



        if( bS[bS_index] == 0 ) {

            continue;

        }



        if( bS[bS_index] < 4 ) {

            const int tc = (tc0_table+52)[index_a][bS[bS_index]] + 1;

            const int p0 = pix[-1];

            const int p1 = pix[-2];

            const int q0 = pix[0];

            const int q1 = pix[1];



            if( FFABS( p0 - q0 ) < alpha &&

                FFABS( p1 - p0 ) < beta &&

                FFABS( q1 - q0 ) < beta ) {

                const int i_delta = av_clip( (((q0 - p0 ) << 2) + (p1 - q1) + 4) >> 3, -tc, tc );



                pix[-1] = av_clip_uint8( p0 + i_delta );    /* p0' */

                pix[0]  = av_clip_uint8( q0 - i_delta );    /* q0' */

                tprintf(h->s.avctx, "filter_mb_mbaff_edgecv i:%d, qp:%d, indexA:%d, alpha:%d, beta:%d, tc:%d\n# bS:%d -> [%02x, %02x, %02x, %02x, %02x, %02x] =>[%02x, %02x, %02x, %02x]\n", i, qp[qp_index], index_a, alpha, beta, tc, bS[bS_index], pix[-3], p1, p0, q0, q1, pix[2], p1, pix[-1], pix[0], q1);

            }

        }else{

            const int p0 = pix[-1];

            const int p1 = pix[-2];

            const int q0 = pix[0];

            const int q1 = pix[1];



            if( FFABS( p0 - q0 ) < alpha &&

                FFABS( p1 - p0 ) < beta &&

                FFABS( q1 - q0 ) < beta ) {



                pix[-1] = ( 2*p1 + p0 + q1 + 2 ) >> 2;   /* p0' */

                pix[0]  = ( 2*q1 + q0 + p1 + 2 ) >> 2;   /* q0' */

                tprintf(h->s.avctx, "filter_mb_mbaff_edgecv i:%d\n# bS:4 -> [%02x, %02x, %02x, %02x, %02x, %02x] =>[%02x, %02x, %02x, %02x, %02x, %02x]\n", i, pix[-3], p1, p0, q0, q1, pix[2], pix[-3], pix[-2], pix[-1], pix[0], pix[1], pix[2]);

            }

        }

    }

}
