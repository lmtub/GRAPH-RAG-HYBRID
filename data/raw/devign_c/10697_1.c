static int mpc7_decode_frame(AVCodecContext * avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MPCContext *c = avctx->priv_data;

    GetBitContext gb;

    uint8_t *bits;

    int i, ch, t;

    int mb = -1;

    Band *bands = c->bands;

    int off;

    int bits_used, bits_avail;



    memset(bands, 0, sizeof(bands));

    if(buf_size <= 4){

        av_log(avctx, AV_LOG_ERROR, "Too small buffer passed (%i bytes)\n", buf_size);

    }



    bits = av_malloc(((buf_size - 1) & ~3) + FF_INPUT_BUFFER_PADDING_SIZE);

    c->dsp.bswap_buf((uint32_t*)bits, (const uint32_t*)(buf + 4), (buf_size - 4) >> 2);

    init_get_bits(&gb, bits, (buf_size - 4)* 8);

    skip_bits(&gb, buf[0]);



    /* read subband indexes */

    for(i = 0; i <= c->maxbands; i++){

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

                bands[i].scf_idx[ch][0] = get_scale_idx(&gb, bands[i].scf_idx[ch][2]);

                switch(bands[i].scfi[ch]){

                case 0:

                    bands[i].scf_idx[ch][1] = get_scale_idx(&gb, bands[i].scf_idx[ch][0]);

                    bands[i].scf_idx[ch][2] = get_scale_idx(&gb, bands[i].scf_idx[ch][1]);

                    break;

                case 1:

                    bands[i].scf_idx[ch][1] = get_scale_idx(&gb, bands[i].scf_idx[ch][0]);

                    bands[i].scf_idx[ch][2] = bands[i].scf_idx[ch][1];

                    break;

                case 2:

                    bands[i].scf_idx[ch][1] = bands[i].scf_idx[ch][0];

                    bands[i].scf_idx[ch][2] = get_scale_idx(&gb, bands[i].scf_idx[ch][1]);

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

    memset(c->Q, 0, sizeof(c->Q));

    off = 0;

    for(i = 0; i < BANDS; i++, off += SAMPLES_PER_BAND)

        for(ch = 0; ch < 2; ch++)

            idx_to_quant(c, &gb, bands[i].res[ch], c->Q[ch] + off);



    ff_mpc_dequantize_and_synth(c, mb, data);



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
