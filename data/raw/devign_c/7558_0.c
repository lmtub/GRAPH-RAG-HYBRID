static int mpc8_decode_frame(AVCodecContext * avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MPCContext *c = avctx->priv_data;

    GetBitContext gb2, *gb = &gb2;

    int i, j, k, ch, cnt, res, t;

    Band *bands = c->bands;

    int off;

    int maxband, keyframe;

    int last[2];



    keyframe = c->cur_frame == 0;



    if(keyframe){

        memset(c->Q, 0, sizeof(c->Q));

        c->last_bits_used = 0;

    }

    init_get_bits(gb, buf, buf_size * 8);

    skip_bits(gb, c->last_bits_used & 7);



    if(keyframe)

        maxband = mpc8_get_mod_golomb(gb, c->maxbands + 1);

    else{

        maxband = c->last_max_band + get_vlc2(gb, band_vlc.table, MPC8_BANDS_BITS, 2);

        if(maxband > 32) maxband -= 33;

    }

    c->last_max_band = maxband;



    /* read subband indexes */

    if(maxband){

        last[0] = last[1] = 0;

        for(i = maxband - 1; i >= 0; i--){

            for(ch = 0; ch < 2; ch++){

                last[ch] = get_vlc2(gb, res_vlc[last[ch] > 2].table, MPC8_RES_BITS, 2) + last[ch];

                if(last[ch] > 15) last[ch] -= 17;

                bands[i].res[ch] = last[ch];

            }

        }

        if(c->MSS){

            int mask;



            cnt = 0;

            for(i = 0; i < maxband; i++)

                if(bands[i].res[0] || bands[i].res[1])

                    cnt++;

            t = mpc8_get_mod_golomb(gb, cnt);

            mask = mpc8_get_mask(gb, cnt, t);

            for(i = maxband - 1; i >= 0; i--)

                if(bands[i].res[0] || bands[i].res[1]){

                    bands[i].msf = mask & 1;

                    mask >>= 1;

                }

        }

    }

    for(i = maxband; i < c->maxbands; i++)

        bands[i].res[0] = bands[i].res[1] = 0;



    if(keyframe){

        for(i = 0; i < 32; i++)

            c->oldDSCF[0][i] = c->oldDSCF[1][i] = 1;

    }



    for(i = 0; i < maxband; i++){

        if(bands[i].res[0] || bands[i].res[1]){

            cnt = !!bands[i].res[0] + !!bands[i].res[1] - 1;

            if(cnt >= 0){

                t = get_vlc2(gb, scfi_vlc[cnt].table, scfi_vlc[cnt].bits, 1);

                if(bands[i].res[0]) bands[i].scfi[0] = t >> (2 * cnt);

                if(bands[i].res[1]) bands[i].scfi[1] = t & 3;

            }

        }

    }



    for(i = 0; i < maxband; i++){

        for(ch = 0; ch < 2; ch++){

            if(!bands[i].res[ch]) continue;



            if(c->oldDSCF[ch][i]){

                bands[i].scf_idx[ch][0] = get_bits(gb, 7) - 6;

                c->oldDSCF[ch][i] = 0;

            }else{

                t = get_vlc2(gb, dscf_vlc[1].table, MPC8_DSCF1_BITS, 2);

                if(t == 64)

                    t += get_bits(gb, 6);

                bands[i].scf_idx[ch][0] = ((bands[i].scf_idx[ch][2] + t - 25) & 0x7F) - 6;

            }

            for(j = 0; j < 2; j++){

                if((bands[i].scfi[ch] << j) & 2)

                    bands[i].scf_idx[ch][j + 1] = bands[i].scf_idx[ch][j];

                else{

                    t = get_vlc2(gb, dscf_vlc[0].table, MPC8_DSCF0_BITS, 2);

                    if(t == 31)

                        t = 64 + get_bits(gb, 6);

                    bands[i].scf_idx[ch][j + 1] = ((bands[i].scf_idx[ch][j] + t - 25) & 0x7F) - 6;

                }

            }

        }

    }



    for(i = 0, off = 0; i < maxband; i++, off += SAMPLES_PER_BAND){

        for(ch = 0; ch < 2; ch++){

            res = bands[i].res[ch];

            switch(res){

            case -1:

                for(j = 0; j < SAMPLES_PER_BAND; j++)

                    c->Q[ch][off + j] = (av_lfg_get(&c->rnd) & 0x3FC) - 510;

                break;

            case 0:

                break;

            case 1:

                for(j = 0; j < SAMPLES_PER_BAND; j += SAMPLES_PER_BAND / 2){

                    cnt = get_vlc2(gb, q1_vlc.table, MPC8_Q1_BITS, 2);

                    t = mpc8_get_mask(gb, 18, cnt);

                    for(k = 0; k < SAMPLES_PER_BAND / 2; k++, t <<= 1)

                        c->Q[ch][off + j + k] = (t & 0x20000) ? (get_bits1(gb) << 1) - 1 : 0;

                }

                break;

            case 2:

                cnt = 6;//2*mpc8_thres[res]

                for(j = 0; j < SAMPLES_PER_BAND; j += 3){

                    t = get_vlc2(gb, q2_vlc[cnt > 3].table, MPC8_Q2_BITS, 2);

                    c->Q[ch][off + j + 0] = mpc8_idx50[t];

                    c->Q[ch][off + j + 1] = mpc8_idx51[t];

                    c->Q[ch][off + j + 2] = mpc8_idx52[t];

                    cnt = (cnt >> 1) + mpc8_huffq2[t];

                }

                break;

            case 3:

            case 4:

                for(j = 0; j < SAMPLES_PER_BAND; j += 2){

                    t = get_vlc2(gb, q3_vlc[res - 3].table, MPC8_Q3_BITS, 2) + q3_offsets[res - 3];

                    c->Q[ch][off + j + 1] = t >> 4;

                    c->Q[ch][off + j + 0] = (t & 8) ? (t & 0xF) - 16 : (t & 0xF);

                }

                break;

            case 5:

            case 6:

            case 7:

            case 8:

                cnt = 2 * mpc8_thres[res];

                for(j = 0; j < SAMPLES_PER_BAND; j++){

                    t = get_vlc2(gb, quant_vlc[res - 5][cnt > mpc8_thres[res]].table, quant_vlc[res - 5][cnt > mpc8_thres[res]].bits, 2) + quant_offsets[res - 5];

                    c->Q[ch][off + j] = t;

                    cnt = (cnt >> 1) + FFABS(c->Q[ch][off + j]);

                }

                break;

            default:

                for(j = 0; j < SAMPLES_PER_BAND; j++){

                    c->Q[ch][off + j] = get_vlc2(gb, q9up_vlc.table, MPC8_Q9UP_BITS, 2);

                    if(res != 9){

                        c->Q[ch][off + j] <<= res - 9;

                        c->Q[ch][off + j] |= get_bits(gb, res - 9);

                    }

                    c->Q[ch][off + j] -= (1 << (res - 2)) - 1;

                }

            }

        }

    }



    ff_mpc_dequantize_and_synth(c, maxband, data, avctx->channels);



    c->cur_frame++;



    c->last_bits_used = get_bits_count(gb);

    if(c->cur_frame >= c->frames)

        c->cur_frame = 0;

    *data_size =  MPC_FRAME_SIZE * 2 * avctx->channels;



    return c->cur_frame ? c->last_bits_used >> 3 : buf_size;

}
