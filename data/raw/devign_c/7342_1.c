static int h261_decode_mb(H261Context *h,

                          DCTELEM block[6][64])

{

    MpegEncContext * const s = &h->s;

    int i, cbp, xy, old_mtype;



    cbp = 63;

    // Read mba

    do{

        h->mba_diff = get_vlc2(&s->gb, h261_mba_vlc.table, H261_MBA_VLC_BITS, 2)+1;

    }

    while( h->mba_diff == MAX_MBA + 1 ); // stuffing



    if ( h->mba_diff < 0 )

        return -1;



    h->current_mba += h->mba_diff;



    if ( h->current_mba > MAX_MBA )

        return -1;

    

    s->mb_x= ((h->gob_number-1) % 2) * 11 + ((h->current_mba-1) % 11);

    s->mb_y= ((h->gob_number-1) / 2) * 3 + ((h->current_mba-1) / 11);



    xy = s->mb_x + s->mb_y * s->mb_stride;



    ff_init_block_index(s);

    ff_update_block_index(s);

    s->dsp.clear_blocks(s->block[0]);



    // Read mtype

    old_mtype = h->mtype;

    h->mtype = get_vlc2(&s->gb, h261_mtype_vlc.table, H261_MTYPE_VLC_BITS, 2);

    h->mtype = h261_mtype_map[h->mtype];



    if (IS_FIL (h->mtype))

        h->loop_filter = 1;



    // Read mquant

    if ( IS_QUANT ( h->mtype ) ){

        ff_set_qscale(s, get_bits(&s->gb, 5));

    }



    s->mb_intra = IS_INTRA4x4(h->mtype);



    // Read mv

    if ( IS_16X16 ( h->mtype ) ){

        // Motion vector data is included for all MC macroblocks. MVD is obtained from the macroblock vector by subtracting the

        // vector of the preceding macroblock. For this calculation the vector of the preceding macroblock is regarded as zero in the

        // following three situations:

        // 1) evaluating MVD for macroblocks 1, 12 and 23;

        // 2) evaluating MVD for macroblocks in which MBA does not represent a difference of 1;

        // 3) MTYPE of the previous macroblock was not MC.

        if ( ( h->current_mba == 1 ) || ( h->current_mba == 12 ) || ( h->current_mba == 23 ) ||

             ( h->mba_diff != 1) || ( !IS_16X16 ( old_mtype ) ))

        {

            h->current_mv_x = 0;

            h->current_mv_y = 0;

        }



        h->current_mv_x= decode_mv_component(&s->gb, h->current_mv_x);

        h->current_mv_y= decode_mv_component(&s->gb, h->current_mv_y);

    }



    // Read cbp

    if ( HAS_CBP( h->mtype ) ){

        cbp = get_vlc2(&s->gb, h261_cbp_vlc.table, H261_CBP_VLC_BITS, 2) + 1;

    }



    if(s->mb_intra){

        s->current_picture.mb_type[xy]= MB_TYPE_INTRA;

        goto intra;

    }



    //set motion vectors

    s->mv_dir = MV_DIR_FORWARD;

    s->mv_type = MV_TYPE_16X16;

    s->current_picture.mb_type[xy]= MB_TYPE_16x16 | MB_TYPE_L0;

    if(IS_16X16 ( h->mtype )){

        s->mv[0][0][0] = h->current_mv_x * 2;//gets divided by 2 in motion compensation

        s->mv[0][0][1] = h->current_mv_y * 2;

    }

    else{

        h->current_mv_x = s->mv[0][0][0] = 0;

        h->current_mv_x = s->mv[0][0][1] = 0;

    }



intra:

    /* decode each block */

    if(s->mb_intra || HAS_CBP(h->mtype)){

        for (i = 0; i < 6; i++) {

            if (h261_decode_block(h, block[i], i, cbp&32) < 0){

                return -1;

            }

            cbp+=cbp;

        }

    }



    /* per-MB end of slice check */

    {

        int v= show_bits(&s->gb, 15);



        if(get_bits_count(&s->gb) + 15 > s->gb.size_in_bits){

            v>>= get_bits_count(&s->gb) + 15 - s->gb.size_in_bits;

        }



        if(v==0){

            return SLICE_END;

        }

    }

    return SLICE_OK;

}
