static void filter_mb_mbaff_edgev( H264Context *h, uint8_t *pix, int stride, int16_t bS[7], int bsi, int qp ) {

    int index_a = qp + h->slice_alpha_c0_offset;

    int alpha = alpha_table[index_a];

    int beta  = beta_table[qp + h->slice_beta_offset];

    if (alpha ==0 || beta == 0) return;



    if( bS[0] < 4 ) {

        int8_t tc[4];

        tc[0] = tc0_table[index_a][bS[0*bsi]];

        tc[1] = tc0_table[index_a][bS[1*bsi]];

        tc[2] = tc0_table[index_a][bS[2*bsi]];

        tc[3] = tc0_table[index_a][bS[3*bsi]];

        h->h264dsp.h264_h_loop_filter_luma_mbaff(pix, stride, alpha, beta, tc);

    } else {

        h->h264dsp.h264_h_loop_filter_luma_mbaff_intra(pix, stride, alpha, beta);

    }

}
