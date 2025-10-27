static int decode_cabac_mb_ref( H264Context *h, int list, int n ) {

    int refa = h->ref_cache[list][scan8[n] - 1];

    int refb = h->ref_cache[list][scan8[n] - 8];

    int ref  = 0;

    int ctx  = 0;



    if( h->slice_type_nos == FF_B_TYPE) {

        if( refa > 0 && !h->direct_cache[scan8[n] - 1] )

            ctx++;

        if( refb > 0 && !h->direct_cache[scan8[n] - 8] )

            ctx += 2;

    } else {

        if( refa > 0 )

            ctx++;

        if( refb > 0 )

            ctx += 2;

    }



    while( get_cabac( &h->cabac, &h->cabac_state[54+ctx] ) ) {

        ref++;

        if( ctx < 4 )

            ctx = 4;

        else

            ctx = 5;

        if(ref >= 32 /*h->ref_list[list]*/){

            av_log(h->s.avctx, AV_LOG_ERROR, "overflow in decode_cabac_mb_ref\n");

            return 0; //FIXME we should return -1 and check the return everywhere

        }

    }

    return ref;

}
