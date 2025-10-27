static int vc1_decode_p_block(VC1Context *v, DCTELEM block[64], int n, int mquant, int ttmb, int first_block,

                              uint8_t *dst, int linesize, int skip_block, int apply_filter, int cbp_top, int cbp_left)

{

    MpegEncContext *s = &v->s;

    GetBitContext *gb = &s->gb;

    int i, j;

    int subblkpat = 0;

    int scale, off, idx, last, skip, value;

    int ttblk = ttmb & 7;

    int pat = 0;



    if(ttmb == -1) {

        ttblk = ff_vc1_ttblk_to_tt[v->tt_index][get_vlc2(gb, ff_vc1_ttblk_vlc[v->tt_index].table, VC1_TTBLK_VLC_BITS, 1)];

    }

    if(ttblk == TT_4X4) {

        subblkpat = ~(get_vlc2(gb, ff_vc1_subblkpat_vlc[v->tt_index].table, VC1_SUBBLKPAT_VLC_BITS, 1) + 1);

    }

    if((ttblk != TT_8X8 && ttblk != TT_4X4) && (v->ttmbf || (ttmb != -1 && (ttmb & 8) && !first_block))) {

        subblkpat = decode012(gb);

        if(subblkpat) subblkpat ^= 3; //swap decoded pattern bits

        if(ttblk == TT_8X4_TOP || ttblk == TT_8X4_BOTTOM) ttblk = TT_8X4;

        if(ttblk == TT_4X8_RIGHT || ttblk == TT_4X8_LEFT) ttblk = TT_4X8;

    }

    scale = 2 * mquant + ((v->pq == mquant) ? v->halfpq : 0);



    // convert transforms like 8X4_TOP to generic TT and SUBBLKPAT

    if(ttblk == TT_8X4_TOP || ttblk == TT_8X4_BOTTOM) {

        subblkpat = 2 - (ttblk == TT_8X4_TOP);

        ttblk = TT_8X4;

    }

    if(ttblk == TT_4X8_RIGHT || ttblk == TT_4X8_LEFT) {

        subblkpat = 2 - (ttblk == TT_4X8_LEFT);

        ttblk = TT_4X8;

    }

    switch(ttblk) {

    case TT_8X8:

        pat = 0xF;

        i = 0;

        last = 0;

        while (!last) {

            vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2);

            i += skip;

            if(i > 63)

                break;

            idx = wmv1_scantable[0][i++];

            block[idx] = value * scale;

            if(!v->pquantizer)

                block[idx] += (block[idx] < 0) ? -mquant : mquant;

        }

        if(!skip_block){

            s->dsp.vc1_inv_trans_8x8(block);

            s->dsp.add_pixels_clamped(block, dst, linesize);

            if(apply_filter && cbp_top  & 0xC)

                vc1_loop_filter(dst, 1, linesize, 8, mquant);

            if(apply_filter && cbp_left & 0xA)

                vc1_loop_filter(dst, linesize, 1, 8, mquant);

        }

        break;

    case TT_4X4:

        pat = ~subblkpat & 0xF;

        for(j = 0; j < 4; j++) {

            last = subblkpat & (1 << (3 - j));

            i = 0;

            off = (j & 1) * 4 + (j & 2) * 16;

            while (!last) {

                vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2);

                i += skip;

                if(i > 15)

                    break;

                idx = ff_vc1_simple_progressive_4x4_zz[i++];

                block[idx + off] = value * scale;

                if(!v->pquantizer)

                    block[idx + off] += (block[idx + off] < 0) ? -mquant : mquant;

            }

            if(!(subblkpat & (1 << (3 - j))) && !skip_block){

                s->dsp.vc1_inv_trans_4x4(dst + (j&1)*4 + (j&2)*2*linesize, linesize, block + off);

                if(apply_filter && (j&2 ? pat & (1<<(j-2)) : (cbp_top & (1 << (j + 2)))))

                    vc1_loop_filter(dst + (j&1)*4 + (j&2)*2*linesize, 1, linesize, 4, mquant);

                if(apply_filter && (j&1 ? pat & (1<<(j-1)) : (cbp_left & (1 << (j + 1)))))

                    vc1_loop_filter(dst + (j&1)*4 + (j&2)*2*linesize, linesize, 1, 4, mquant);

            }

        }

        break;

    case TT_8X4:

        pat = ~((subblkpat & 2)*6 + (subblkpat & 1)*3) & 0xF;

        for(j = 0; j < 2; j++) {

            last = subblkpat & (1 << (1 - j));

            i = 0;

            off = j * 32;

            while (!last) {

                vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2);

                i += skip;

                if(i > 31)

                    break;

                idx = v->zz_8x4[i++]+off;

                block[idx] = value * scale;

                if(!v->pquantizer)

                    block[idx] += (block[idx] < 0) ? -mquant : mquant;

            }

            if(!(subblkpat & (1 << (1 - j))) && !skip_block){

                s->dsp.vc1_inv_trans_8x4(dst + j*4*linesize, linesize, block + off);

                if(apply_filter && j ? pat & 0x3 : (cbp_top & 0xC))

                    vc1_loop_filter(dst + j*4*linesize, 1, linesize, 8, mquant);

                if(apply_filter && cbp_left & (2 << j))

                    vc1_loop_filter(dst + j*4*linesize, linesize, 1, 4, mquant);

            }

        }

        break;

    case TT_4X8:

        pat = ~(subblkpat*5) & 0xF;

        for(j = 0; j < 2; j++) {

            last = subblkpat & (1 << (1 - j));

            i = 0;

            off = j * 4;

            while (!last) {

                vc1_decode_ac_coeff(v, &last, &skip, &value, v->codingset2);

                i += skip;

                if(i > 31)

                    break;

                idx = v->zz_4x8[i++]+off;

                block[idx] = value * scale;

                if(!v->pquantizer)

                    block[idx] += (block[idx] < 0) ? -mquant : mquant;

            }

            if(!(subblkpat & (1 << (1 - j))) && !skip_block){

                s->dsp.vc1_inv_trans_4x8(dst + j*4, linesize, block + off);

                if(apply_filter && cbp_top & (2 << j))

                    vc1_loop_filter(dst + j*4, 1, linesize, 4, mquant);

                if(apply_filter && j ? pat & 0x5 : (cbp_left & 0xA))

                    vc1_loop_filter(dst + j*4, linesize, 1, 8, mquant);

            }

        }

        break;

    }

    return pat;

}
