static int alloc_tables(H264Context *h){

    MpegEncContext * const s = &h->s;

    const int big_mb_num= s->mb_stride * (s->mb_height+1);

    int x,y;



    CHECKED_ALLOCZ(h->intra4x4_pred_mode, big_mb_num * 8  * sizeof(uint8_t))



    CHECKED_ALLOCZ(h->non_zero_count    , big_mb_num * 16 * sizeof(uint8_t))

    CHECKED_ALLOCZ(h->slice_table_base  , (big_mb_num+s->mb_stride) * sizeof(*h->slice_table_base))

    CHECKED_ALLOCZ(h->cbp_table, big_mb_num * sizeof(uint16_t))



    CHECKED_ALLOCZ(h->chroma_pred_mode_table, big_mb_num * sizeof(uint8_t))

    CHECKED_ALLOCZ(h->mvd_table[0], 32*big_mb_num * sizeof(uint16_t));

    CHECKED_ALLOCZ(h->mvd_table[1], 32*big_mb_num * sizeof(uint16_t));

    CHECKED_ALLOCZ(h->direct_table, 32*big_mb_num * sizeof(uint8_t));



    memset(h->slice_table_base, -1, (big_mb_num+s->mb_stride)  * sizeof(*h->slice_table_base));

    h->slice_table= h->slice_table_base + s->mb_stride*2 + 1;



    CHECKED_ALLOCZ(h->mb2b_xy  , big_mb_num * sizeof(uint32_t));

    CHECKED_ALLOCZ(h->mb2b8_xy , big_mb_num * sizeof(uint32_t));

    for(y=0; y<s->mb_height; y++){

        for(x=0; x<s->mb_width; x++){

            const int mb_xy= x + y*s->mb_stride;

            const int b_xy = 4*x + 4*y*h->b_stride;

            const int b8_xy= 2*x + 2*y*h->b8_stride;



            h->mb2b_xy [mb_xy]= b_xy;

            h->mb2b8_xy[mb_xy]= b8_xy;

        }

    }



    s->obmc_scratchpad = NULL;



    if(!h->dequant4_coeff[0])

        init_dequant_tables(h);



    return 0;

fail:

    free_tables(h);

    return -1;

}
