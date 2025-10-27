static int decode_cabac_mb_cbp_luma( H264Context *h) {

    int cbp_b, cbp_a, ctx, cbp = 0;



    cbp_a = h->slice_table[h->left_mb_xy[0]] == h->slice_num ? h->left_cbp : -1;

    cbp_b = h->slice_table[h->top_mb_xy]     == h->slice_num ? h->top_cbp  : -1;



    ctx = !(cbp_a & 0x02) + 2 * !(cbp_b & 0x04);

    cbp |= get_cabac_noinline(&h->cabac, &h->cabac_state[73 + ctx]);

    ctx = !(cbp   & 0x01) + 2 * !(cbp_b & 0x08);

    cbp |= get_cabac_noinline(&h->cabac, &h->cabac_state[73 + ctx]) << 1;

    ctx = !(cbp_a & 0x08) + 2 * !(cbp   & 0x01);

    cbp |= get_cabac_noinline(&h->cabac, &h->cabac_state[73 + ctx]) << 2;

    ctx = !(cbp   & 0x04) + 2 * !(cbp   & 0x02);

    cbp |= get_cabac_noinline(&h->cabac, &h->cabac_state[73 + ctx]) << 3;

    return cbp;

}
