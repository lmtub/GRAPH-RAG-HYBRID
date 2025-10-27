static int mpc8_decode_init(AVCodecContext * avctx)

{

    int i;

    MPCContext *c = avctx->priv_data;

    GetBitContext gb;

    static int vlc_inited = 0;



    if(avctx->extradata_size < 2){

        av_log(avctx, AV_LOG_ERROR, "Too small extradata size (%i)!\n", avctx->extradata_size);

        return -1;

    }

    memset(c->oldDSCF, 0, sizeof(c->oldDSCF));

    av_init_random(0xDEADBEEF, &c->rnd);

    dsputil_init(&c->dsp, avctx);



    ff_mpc_init();



    init_get_bits(&gb, avctx->extradata, 16);



    skip_bits(&gb, 3);//sample rate

    c->maxbands = get_bits(&gb, 5) + 1;

    skip_bits(&gb, 4);//channels

    c->MSS = get_bits1(&gb);

    c->frames = 1 << (get_bits(&gb, 3) * 2);



    if(vlc_inited) return 0;

    av_log(avctx, AV_LOG_DEBUG, "Initing VLC\n");



    init_vlc(&band_vlc, MPC8_BANDS_BITS, MPC8_BANDS_SIZE,

             mpc8_bands_bits,  1, 1,

             mpc8_bands_codes, 1, 1, INIT_VLC_USE_STATIC);



    init_vlc(&q1_vlc, MPC8_Q1_BITS, MPC8_Q1_SIZE,

             mpc8_q1_bits,  1, 1,

             mpc8_q1_codes, 1, 1, INIT_VLC_USE_STATIC);

    init_vlc(&q9up_vlc, MPC8_Q9UP_BITS, MPC8_Q9UP_SIZE,

             mpc8_q9up_bits,  1, 1,

             mpc8_q9up_codes, 1, 1, INIT_VLC_USE_STATIC);



    init_vlc(&scfi_vlc[0], MPC8_SCFI0_BITS, MPC8_SCFI0_SIZE,

             mpc8_scfi0_bits,  1, 1,

             mpc8_scfi0_codes, 1, 1, INIT_VLC_USE_STATIC);

    init_vlc(&scfi_vlc[1], MPC8_SCFI1_BITS, MPC8_SCFI1_SIZE,

             mpc8_scfi1_bits,  1, 1,

             mpc8_scfi1_codes, 1, 1, INIT_VLC_USE_STATIC);



    init_vlc(&dscf_vlc[0], MPC8_DSCF0_BITS, MPC8_DSCF0_SIZE,

             mpc8_dscf0_bits,  1, 1,

             mpc8_dscf0_codes, 1, 1, INIT_VLC_USE_STATIC);

    init_vlc(&dscf_vlc[1], MPC8_DSCF1_BITS, MPC8_DSCF1_SIZE,

             mpc8_dscf1_bits,  1, 1,

             mpc8_dscf1_codes, 1, 1, INIT_VLC_USE_STATIC);



    init_vlc_sparse(&q3_vlc[0], MPC8_Q3_BITS, MPC8_Q3_SIZE,

             mpc8_q3_bits,  1, 1,

             mpc8_q3_codes, 1, 1,

             mpc8_q3_syms,  1, 1, INIT_VLC_USE_STATIC);

    init_vlc_sparse(&q3_vlc[1], MPC8_Q4_BITS, MPC8_Q4_SIZE,

             mpc8_q4_bits,  1, 1,

             mpc8_q4_codes, 1, 1,

             mpc8_q4_syms,  1, 1, INIT_VLC_USE_STATIC);



    for(i = 0; i < 2; i++){

        init_vlc(&res_vlc[i], MPC8_RES_BITS, MPC8_RES_SIZE,

                 &mpc8_res_bits[i],  1, 1,

                 &mpc8_res_codes[i], 1, 1, INIT_VLC_USE_STATIC);



        init_vlc(&q2_vlc[i], MPC8_Q2_BITS, MPC8_Q2_SIZE,

                 &mpc8_q2_bits[i],  1, 1,

                 &mpc8_q2_codes[i], 1, 1, INIT_VLC_USE_STATIC);



        init_vlc(&quant_vlc[0][i], MPC8_Q5_BITS, MPC8_Q5_SIZE,

                 &mpc8_q5_bits[i],  1, 1,

                 &mpc8_q5_codes[i], 1, 1, INIT_VLC_USE_STATIC);

        init_vlc(&quant_vlc[1][i], MPC8_Q6_BITS, MPC8_Q6_SIZE,

                 &mpc8_q6_bits[i],  1, 1,

                 &mpc8_q6_codes[i], 1, 1, INIT_VLC_USE_STATIC);

        init_vlc(&quant_vlc[2][i], MPC8_Q7_BITS, MPC8_Q7_SIZE,

                 &mpc8_q7_bits[i],  1, 1,

                 &mpc8_q7_codes[i], 1, 1, INIT_VLC_USE_STATIC);

        init_vlc(&quant_vlc[3][i], MPC8_Q8_BITS, MPC8_Q8_SIZE,

                 &mpc8_q8_bits[i],  1, 1,

                 &mpc8_q8_codes[i], 1, 1, INIT_VLC_USE_STATIC);

    }

    vlc_inited = 1;

    return 0;

}
