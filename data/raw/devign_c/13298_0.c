static int mpc7_decode_init(AVCodecContext * avctx)

{

    int i, j;

    MPCContext *c = avctx->priv_data;

    GetBitContext gb;

    uint8_t buf[16];

    static int vlc_inited = 0;



    if(avctx->extradata_size < 16){

        av_log(avctx, AV_LOG_ERROR, "Too small extradata size (%i)!\n", avctx->extradata_size);

        return -1;

    }

    memset(c->oldDSCF, 0, sizeof(c->oldDSCF));

    av_init_random(0xDEADBEEF, &c->rnd);

    dsputil_init(&c->dsp, avctx);

    c->dsp.bswap_buf((uint32_t*)buf, (const uint32_t*)avctx->extradata, 4);

    ff_mpc_init();

    init_get_bits(&gb, buf, 128);



    c->IS = get_bits1(&gb);

    c->MSS = get_bits1(&gb);

    c->maxbands = get_bits(&gb, 6);

    if(c->maxbands >= BANDS){

        av_log(avctx, AV_LOG_ERROR, "Too many bands: %i\n", c->maxbands);

        return -1;

    }

    skip_bits(&gb, 88);

    c->gapless = get_bits1(&gb);

    c->lastframelen = get_bits(&gb, 11);

    av_log(avctx, AV_LOG_DEBUG, "IS: %d, MSS: %d, TG: %d, LFL: %d, bands: %d\n",

            c->IS, c->MSS, c->gapless, c->lastframelen, c->maxbands);

    c->frames_to_skip = 0;



    if(vlc_inited) return 0;

    av_log(avctx, AV_LOG_DEBUG, "Initing VLC\n");

    if(init_vlc(&scfi_vlc, MPC7_SCFI_BITS, MPC7_SCFI_SIZE,

                &mpc7_scfi[1], 2, 1,

                &mpc7_scfi[0], 2, 1, INIT_VLC_USE_STATIC)){

        av_log(avctx, AV_LOG_ERROR, "Cannot init SCFI VLC\n");

        return -1;

    }

    if(init_vlc(&dscf_vlc, MPC7_DSCF_BITS, MPC7_DSCF_SIZE,

                &mpc7_dscf[1], 2, 1,

                &mpc7_dscf[0], 2, 1, INIT_VLC_USE_STATIC)){

        av_log(avctx, AV_LOG_ERROR, "Cannot init DSCF VLC\n");

        return -1;

    }

    if(init_vlc(&hdr_vlc, MPC7_HDR_BITS, MPC7_HDR_SIZE,

                &mpc7_hdr[1], 2, 1,

                &mpc7_hdr[0], 2, 1, INIT_VLC_USE_STATIC)){

        av_log(avctx, AV_LOG_ERROR, "Cannot init HDR VLC\n");

        return -1;

    }

    for(i = 0; i < MPC7_QUANT_VLC_TABLES; i++){

        for(j = 0; j < 2; j++){

            if(init_vlc(&quant_vlc[i][j], 9, mpc7_quant_vlc_sizes[i],

                        &mpc7_quant_vlc[i][j][1], 4, 2,

                        &mpc7_quant_vlc[i][j][0], 4, 2, INIT_VLC_USE_STATIC)){

                av_log(avctx, AV_LOG_ERROR, "Cannot init QUANT VLC %i,%i\n",i,j);

                return -1;

            }

        }

    }

    vlc_inited = 1;

    return 0;

}
