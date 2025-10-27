static int rv40_decode_mb_info(RV34DecContext *r)

{

    MpegEncContext *s = &r->s;

    GetBitContext *gb = &s->gb;

    int q, i;

    int prev_type = 0;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;

    int blocks[RV34_MB_TYPES] = {0};

    int count = 0;



    if(!r->s.mb_skip_run)

        r->s.mb_skip_run = svq3_get_ue_golomb(gb) + 1;



    if(--r->s.mb_skip_run)

         return RV34_MB_SKIP;



    if(r->avail_cache[6-1])

        blocks[r->mb_type[mb_pos - 1]]++;

    if(r->avail_cache[6-4]){

        blocks[r->mb_type[mb_pos - s->mb_stride]]++;

        if(r->avail_cache[6-2])

            blocks[r->mb_type[mb_pos - s->mb_stride + 1]]++;

        if(r->avail_cache[6-5])

            blocks[r->mb_type[mb_pos - s->mb_stride - 1]]++;

    }



    for(i = 0; i < RV34_MB_TYPES; i++){

        if(blocks[i] > count){

            count = blocks[i];

            prev_type = i;

        }

    }

    if(s->pict_type == AV_PICTURE_TYPE_P){

        prev_type = block_num_to_ptype_vlc_num[prev_type];

        q = get_vlc2(gb, ptype_vlc[prev_type].table, PTYPE_VLC_BITS, 1);

        if(q < PBTYPE_ESCAPE)

            return q;

        q = get_vlc2(gb, ptype_vlc[prev_type].table, PTYPE_VLC_BITS, 1);

        av_log(s->avctx, AV_LOG_ERROR, "Dquant for P-frame\n");

    }else{

        prev_type = block_num_to_btype_vlc_num[prev_type];

        q = get_vlc2(gb, btype_vlc[prev_type].table, BTYPE_VLC_BITS, 1);

        if(q < PBTYPE_ESCAPE)

            return q;

        q = get_vlc2(gb, btype_vlc[prev_type].table, BTYPE_VLC_BITS, 1);

        av_log(s->avctx, AV_LOG_ERROR, "Dquant for B-frame\n");

    }

    return 0;

}
