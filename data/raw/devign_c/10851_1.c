static int mpc7_decode_frame(AVCodecContext * avctx,

                            void *data, int *data_size,

                            uint8_t * buf, int buf_size)

{

    MPCContext *c = avctx->priv_data;

    GetBitContext gb;

    uint8_t *bits;

    int i, j, ch, t;

    int mb = -1;

    Band bands[BANDS];

    int Q[2][MPC_FRAME_SIZE];

    int off;

    float mul;

    int bits_used, bits_avail;



    memset(bands, 0, sizeof(bands));

    if(buf_size <= 4){

        av_log(avctx, AV_LOG_ERROR, "Too small buffer passed (%i bytes)\n", buf_size);

    }



    bits = av_malloc((buf_size - 1) & ~3);

    c->dsp.bswap_buf(bits, buf + 4, (buf_size - 4) >> 2);

    init_get_bits(&gb, bits, (buf_size - 4)* 8);

    skip_bits(&gb, buf[0]);



    /* read subband indexes */

    for(i = 0; i <= c->bands; i++){

        for(ch = 0; ch < 2; ch++){

            if(i) t = get_vlc2(&gb, hdr_vlc.table, MPC7_HDR_BITS, 1) - 5;

            if(!i || (t == 4)) bands[i].res[ch] = get_bits(&gb, 4);

            else bands[i].res[ch] = bands[i-1].res[ch] + t;

        }



        if(bands[i].res[0] || bands[i].res[1]){

            mb = i;

            if(c->MSS) bands[i].msf = get_bits1(&gb);

        }

    }

    /* get scale indexes coding method */

    for(i = 0; i <= mb; i++)

        for(ch = 0; ch < 2; ch++)

            if(bands[i].res[ch]) bands[i].scfi[ch] = get_vlc2(&gb, scfi_vlc.table, MPC7_SCFI_BITS, 1);

    /* get scale indexes */

    for(i = 0; i <= mb; i++){

        for(ch = 0; ch < 2; ch++){

            if(bands[i].res[ch]){

                bands[i].scf_idx[ch][2] = c->oldDSCF[ch][i];

                t = get_vlc2(&gb, dscf_vlc.table, MPC7_DSCF_BITS, 1) - 7;

                bands[i].scf_idx[ch][0] = (t == 8) ? get_bits(&gb, 6) : (bands[i].scf_idx[ch][2] + t);

                switch(bands[i].scfi[ch]){

                case 0:

                    t = get_vlc2(&gb, dscf_vlc.table, MPC7_DSCF_BITS, 1) - 7;

                    bands[i].scf_idx[ch][1] = (t == 8) ? get_bits(&gb, 6) : (bands[i].scf_idx[ch][0] + t);

                    t = get_vlc2(&gb, dscf_vlc.table, MPC7_DSCF_BITS, 1) - 7;

                    bands[i].scf_idx[ch][2] = (t == 8) ? get_bits(&gb, 6) : (bands[i].scf_idx[ch][1] + t);

                    break;

                case 1:

                    t = get_vlc2(&gb, dscf_vlc.table, MPC7_DSCF_BITS, 1) - 7;

                    bands[i].scf_idx[ch][1] = (t == 8) ? get_bits(&gb, 6) : (bands[i].scf_idx[ch][0] + t);

                    bands[i].scf_idx[ch][2] = bands[i].scf_idx[ch][1];

                    break;

                case 2:

                    bands[i].scf_idx[ch][1] = bands[i].scf_idx[ch][0];

                    t = get_vlc2(&gb, dscf_vlc.table, MPC7_DSCF_BITS, 1) - 7;

                    bands[i].scf_idx[ch][2] = (t == 8) ? get_bits(&gb, 6) : (bands[i].scf_idx[ch][1] + t);

                    break;

                case 3:

                    bands[i].scf_idx[ch][2] = bands[i].scf_idx[ch][1] = bands[i].scf_idx[ch][0];

                    break;

                }

                c->oldDSCF[ch][i] = bands[i].scf_idx[ch][2];

            }

        }

    }

    /* get quantizers */

    memset(Q, 0, sizeof(Q));

    off = 0;

    for(i = 0; i < BANDS; i++, off += SAMPLES_PER_BAND)

        for(ch = 0; ch < 2; ch++)

            idx_to_quant(c, &gb, bands[i].res[ch], Q[ch] + off);

    /* dequantize */

    memset(c->sb_samples, 0, sizeof(c->sb_samples));

    off = 0;

    for(i = 0; i <= mb; i++, off += SAMPLES_PER_BAND){

        for(ch = 0; ch < 2; ch++){

            if(bands[i].res[ch]){

                j = 0;

                mul = mpc_CC[bands[i].res[ch]] * mpc7_SCF[bands[i].scf_idx[ch][0]];

                for(; j < 12; j++)

                    c->sb_samples[ch][j][i] = mul * Q[ch][j + off];

                mul = mpc_CC[bands[i].res[ch]] * mpc7_SCF[bands[i].scf_idx[ch][1]];

                for(; j < 24; j++)

                    c->sb_samples[ch][j][i] = mul * Q[ch][j + off];

                mul = mpc_CC[bands[i].res[ch]] * mpc7_SCF[bands[i].scf_idx[ch][2]];

                for(; j < 36; j++)

                    c->sb_samples[ch][j][i] = mul * Q[ch][j + off];

            }

        }

        if(bands[i].msf){

            int t1, t2;

            for(j = 0; j < SAMPLES_PER_BAND; j++){

                t1 = c->sb_samples[0][j][i];

                t2 = c->sb_samples[1][j][i];

                c->sb_samples[0][j][i] = t1 + t2;

                c->sb_samples[1][j][i] = t1 - t2;

            }

        }

    }



    mpc_synth(c, data);



    av_free(bits);



    bits_used = get_bits_count(&gb);

    bits_avail = (buf_size - 4) * 8;

    if(!buf[1] && ((bits_avail < bits_used) || (bits_used + 32 <= bits_avail))){

        av_log(NULL,0, "Error decoding frame: used %i of %i bits\n", bits_used, bits_avail);

        return -1;

    }

    if(c->frames_to_skip){

        c->frames_to_skip--;

        *data_size = 0;

        return buf_size;

    }

    *data_size = (buf[1] ? c->lastframelen : MPC_FRAME_SIZE) * 4;



    return buf_size;

}
