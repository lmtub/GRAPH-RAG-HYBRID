static int vorbis_parse_id_hdr(vorbis_context *vc){

    GetBitContext *gb=&vc->gb;

    uint_fast8_t bl0, bl1;



    if ((get_bits(gb, 8)!='v') || (get_bits(gb, 8)!='o') ||

    (get_bits(gb, 8)!='r') || (get_bits(gb, 8)!='b') ||

    (get_bits(gb, 8)!='i') || (get_bits(gb, 8)!='s')) {

        av_log(vc->avccontext, AV_LOG_ERROR, " Vorbis id header packet corrupt (no vorbis signature). \n");

        return 1;

    }



    vc->version=get_bits_long(gb, 32);    //FIXME check 0

    vc->audio_channels=get_bits(gb, 8);   //FIXME check >0

    vc->audio_samplerate=get_bits_long(gb, 32);   //FIXME check >0

    vc->bitrate_maximum=get_bits_long(gb, 32);

    vc->bitrate_nominal=get_bits_long(gb, 32);

    vc->bitrate_minimum=get_bits_long(gb, 32);

    bl0=get_bits(gb, 4);

    bl1=get_bits(gb, 4);

    vc->blocksize[0]=(1<<bl0);

    vc->blocksize[1]=(1<<bl1);

    if (bl0>13 || bl0<6 || bl1>13 || bl1<6 || bl1<bl0) {

        av_log(vc->avccontext, AV_LOG_ERROR, " Vorbis id header packet corrupt (illegal blocksize). \n");

        return 3;

    }

    // output format int16

    if (vc->blocksize[1]/2 * vc->audio_channels * 2 >

                                             AVCODEC_MAX_AUDIO_FRAME_SIZE) {

        av_log(vc->avccontext, AV_LOG_ERROR, "Vorbis channel count makes "

               "output packets too large.\n");

        return 4;

    }

    vc->win[0]=ff_vorbis_vwin[bl0-6];

    vc->win[1]=ff_vorbis_vwin[bl1-6];



    if(vc->exp_bias){

        int i, j;

        for(j=0; j<2; j++){

            float *win = av_malloc(vc->blocksize[j]/2 * sizeof(float));

            for(i=0; i<vc->blocksize[j]/2; i++)

                win[i] = vc->win[j][i] * (1<<15);

            vc->win[j] = win;

        }

    }



    if ((get_bits1(gb)) == 0) {

        av_log(vc->avccontext, AV_LOG_ERROR, " Vorbis id header packet corrupt (framing flag not set). \n");

        return 2;

    }



    vc->channel_residues= av_malloc((vc->blocksize[1]/2)*vc->audio_channels * sizeof(float));

    vc->channel_floors  = av_malloc((vc->blocksize[1]/2)*vc->audio_channels * sizeof(float));

    vc->saved           = av_mallocz((vc->blocksize[1]/2)*vc->audio_channels * sizeof(float));

    vc->ret             = av_malloc((vc->blocksize[1]/2)*vc->audio_channels * sizeof(float));

    vc->buf             = av_malloc( vc->blocksize[1]                       * sizeof(float));

    vc->buf_tmp         = av_malloc( vc->blocksize[1]                       * sizeof(float));

    vc->previous_window=0;



    ff_mdct_init(&vc->mdct[0], bl0, 1);

    ff_mdct_init(&vc->mdct[1], bl1, 1);



    AV_DEBUG(" vorbis version %d \n audio_channels %d \n audio_samplerate %d \n bitrate_max %d \n bitrate_nom %d \n bitrate_min %d \n blk_0 %d blk_1 %d \n ",

            vc->version, vc->audio_channels, vc->audio_samplerate, vc->bitrate_maximum, vc->bitrate_nominal, vc->bitrate_minimum, vc->blocksize[0], vc->blocksize[1]);



/*

    BLK=vc->blocksize[0];

    for(i=0;i<BLK/2;++i) {

        vc->win[0][i]=sin(0.5*3.14159265358*(sin(((float)i+0.5)/(float)BLK*3.14159265358))*(sin(((float)i+0.5)/(float)BLK*3.14159265358)));

    }

*/



    return 0;

}
