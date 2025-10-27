static inline int get_cabac_cbf_ctx( H264Context *h, int cat, int idx ) {

    int nza, nzb;

    int ctx = 0;



    if( cat == 0 ) {

        nza = h->left_cbp&0x100;

        nzb = h-> top_cbp&0x100;

    } else if( cat == 1 || cat == 2 ) {

        nza = h->non_zero_count_cache[scan8[idx] - 1];

        nzb = h->non_zero_count_cache[scan8[idx] - 8];

    } else if( cat == 3 ) {

        nza = (h->left_cbp>>(6+idx))&0x01;

        nzb = (h-> top_cbp>>(6+idx))&0x01;

    } else {

        assert(cat == 4);

        nza = h->non_zero_count_cache[scan8[16+idx] - 1];

        nzb = h->non_zero_count_cache[scan8[16+idx] - 8];

    }



    if( nza > 0 )

        ctx++;



    if( nzb > 0 )

        ctx += 2;



    return ctx + 4 * cat;

}
