static int rv34_decode_macroblock(RV34DecContext *r, int8_t *intra_types)

{

    MpegEncContext *s = &r->s;

    GetBitContext *gb = &s->gb;

    int cbp, cbp2;

    int i, blknum, blkoff;

    DCTELEM block16[64];

    int luma_dc_quant;

    int dist;

    int mb_pos = s->mb_x + s->mb_y * s->mb_stride;



    // Calculate which neighbours are available. Maybe it's worth optimizing too.

    memset(r->avail_cache, 0, sizeof(r->avail_cache));

    fill_rectangle(r->avail_cache + 5, 2, 2, 4, 1, 4);

    dist = (s->mb_x - s->resync_mb_x) + (s->mb_y - s->resync_mb_y) * s->mb_width;

    if(s->mb_x && dist)

        r->avail_cache[4] =

        r->avail_cache[8] = s->current_picture_ptr->mb_type[mb_pos - 1];

    if(dist >= s->mb_width)

        r->avail_cache[1] =

        r->avail_cache[2] = s->current_picture_ptr->mb_type[mb_pos - s->mb_stride];

    if(((s->mb_x+1) < s->mb_width) && dist >= s->mb_width - 1)

        r->avail_cache[3] = s->current_picture_ptr->mb_type[mb_pos - s->mb_stride + 1];

    if(s->mb_x && dist > s->mb_width)

        r->avail_cache[0] = s->current_picture_ptr->mb_type[mb_pos - s->mb_stride - 1];



    s->qscale = r->si.quant;

    cbp = cbp2 = rv34_decode_mb_header(r, intra_types);

    r->cbp_luma  [s->mb_x + s->mb_y * s->mb_stride] = cbp;

    r->cbp_chroma[s->mb_x + s->mb_y * s->mb_stride] = cbp >> 16;

    if(s->pict_type == FF_I_TYPE)

        r->deblock_coefs[mb_pos] = 0;

    else

        r->deblock_coefs[mb_pos] = rv34_set_deblock_coef(r);

    s->current_picture_ptr->qscale_table[s->mb_x + s->mb_y * s->mb_stride] = s->qscale;



    if(cbp == -1)

        return -1;



    luma_dc_quant = r->block_type == RV34_MB_P_MIX16x16 ? r->luma_dc_quant_p[s->qscale] : r->luma_dc_quant_i[s->qscale];

    if(r->is16){

        memset(block16, 0, sizeof(block16));

        rv34_decode_block(block16, gb, r->cur_vlcs, 3, 0);

        rv34_dequant4x4_16x16(block16, rv34_qscale_tab[luma_dc_quant],rv34_qscale_tab[s->qscale]);

        rv34_inv_transform_noround(block16);

    }



    for(i = 0; i < 16; i++, cbp >>= 1){

        if(!r->is16 && !(cbp & 1)) continue;

        blknum = ((i & 2) >> 1) + ((i & 8) >> 2);

        blkoff = ((i & 1) << 2) + ((i & 4) << 3);

        if(cbp & 1)

            rv34_decode_block(s->block[blknum] + blkoff, gb, r->cur_vlcs, r->luma_vlc, 0);

        rv34_dequant4x4(s->block[blknum] + blkoff, rv34_qscale_tab[s->qscale],rv34_qscale_tab[s->qscale]);

        if(r->is16) //FIXME: optimize

            s->block[blknum][blkoff] = block16[(i & 3) | ((i & 0xC) << 1)];

        rv34_inv_transform(s->block[blknum] + blkoff);

    }

    if(r->block_type == RV34_MB_P_MIX16x16)

        r->cur_vlcs = choose_vlc_set(r->si.quant, r->si.vlc_set, 1);

    for(; i < 24; i++, cbp >>= 1){

        if(!(cbp & 1)) continue;

        blknum = ((i & 4) >> 2) + 4;

        blkoff = ((i & 1) << 2) + ((i & 2) << 4);

        rv34_decode_block(s->block[blknum] + blkoff, gb, r->cur_vlcs, r->chroma_vlc, 1);

        rv34_dequant4x4(s->block[blknum] + blkoff, rv34_qscale_tab[rv34_chroma_quant[1][s->qscale]],rv34_qscale_tab[rv34_chroma_quant[0][s->qscale]]);

        rv34_inv_transform(s->block[blknum] + blkoff);

    }

    if(IS_INTRA(s->current_picture_ptr->mb_type[s->mb_x + s->mb_y*s->mb_stride]))

        rv34_output_macroblock(r, intra_types, cbp2, r->is16);

    else

        rv34_apply_differences(r, cbp2);



    return 0;

}
