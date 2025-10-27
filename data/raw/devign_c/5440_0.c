static av_cold int aac_decode_init(AVCodecContext * avccontext) {

    AACContext * ac = avccontext->priv_data;

    int i;



    ac->avccontext = avccontext;



    if (avccontext->extradata_size <= 0 ||

        decode_audio_specific_config(ac, avccontext->extradata, avccontext->extradata_size))

        return -1;



    avccontext->sample_fmt  = SAMPLE_FMT_S16;

    avccontext->sample_rate = ac->m4ac.sample_rate;

    avccontext->frame_size  = 1024;



    AAC_INIT_VLC_STATIC( 0, 144);

    AAC_INIT_VLC_STATIC( 1, 114);

    AAC_INIT_VLC_STATIC( 2, 188);

    AAC_INIT_VLC_STATIC( 3, 180);

    AAC_INIT_VLC_STATIC( 4, 172);

    AAC_INIT_VLC_STATIC( 5, 140);

    AAC_INIT_VLC_STATIC( 6, 168);

    AAC_INIT_VLC_STATIC( 7, 114);

    AAC_INIT_VLC_STATIC( 8, 262);

    AAC_INIT_VLC_STATIC( 9, 248);

    AAC_INIT_VLC_STATIC(10, 384);



    dsputil_init(&ac->dsp, avccontext);



    ac->random_state = 0x1f2e3d4c;



    // -1024 - Compensate wrong IMDCT method.

    // 32768 - Required to scale values to the correct range for the bias method

    //         for float to int16 conversion.



    if(ac->dsp.float_to_int16 == ff_float_to_int16_c) {

        ac->add_bias = 385.0f;

        ac->sf_scale = 1. / (-1024. * 32768.);

        ac->sf_offset = 0;

    } else {

        ac->add_bias = 0.0f;

        ac->sf_scale = 1. / -1024.;

        ac->sf_offset = 60;

    }



#ifndef CONFIG_HARDCODED_TABLES

    for (i = 0; i < 428; i++)

        ff_aac_pow2sf_tab[i] = pow(2, (i - 200)/4.);

#endif /* CONFIG_HARDCODED_TABLES */



    INIT_VLC_STATIC(&vlc_scalefactors,7,FF_ARRAY_ELEMS(ff_aac_scalefactor_code),

        ff_aac_scalefactor_bits, sizeof(ff_aac_scalefactor_bits[0]), sizeof(ff_aac_scalefactor_bits[0]),

        ff_aac_scalefactor_code, sizeof(ff_aac_scalefactor_code[0]), sizeof(ff_aac_scalefactor_code[0]),

        352);



    ff_mdct_init(&ac->mdct, 11, 1);

    ff_mdct_init(&ac->mdct_small, 8, 1);

    // window initialization

    ff_kbd_window_init(ff_aac_kbd_long_1024, 4.0, 1024);

    ff_kbd_window_init(ff_aac_kbd_short_128, 6.0, 128);

    ff_sine_window_init(ff_sine_1024, 1024);

    ff_sine_window_init(ff_sine_128, 128);



    return 0;

}
