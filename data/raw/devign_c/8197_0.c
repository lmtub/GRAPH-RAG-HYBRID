static void av_noinline filter_mb_edgev( uint8_t *pix, int stride, int16_t bS[4], unsigned int qp, H264Context *h) {

    const unsigned int index_a = 52 + qp + h->slice_alpha_c0_offset;

    const int alpha = alpha_table[index_a];

    const int beta  = (beta_table+52)[qp + h->slice_beta_offset];

    if (alpha ==0 || beta == 0) return;



    if( bS[0] < 4 ) {

        int8_t tc[4];

        tc[0] = tc0_table[index_a][bS[0]];

        tc[1] = tc0_table[index_a][bS[1]];

        tc[2] = tc0_table[index_a][bS[2]];

        tc[3] = tc0_table[index_a][bS[3]];

        h->s.dsp.h264_h_loop_filter_luma(pix, stride, alpha, beta, tc);

    } else {

        h->s.dsp.h264_h_loop_filter_luma_intra(pix, stride, alpha, beta);

    }

}
