static av_cold void init_atrac3_transforms(ATRAC3Context *q) {

    float enc_window[256];

    int i;



    /* Generate the mdct window, for details see

     * http://wiki.multimedia.cx/index.php?title=RealAudio_atrc#Windows */

    for (i=0 ; i<256; i++)

        enc_window[i] = (sin(((i + 0.5) / 256.0 - 0.5) * M_PI) + 1.0) * 0.5;



    if (!mdct_window[0])

        for (i=0 ; i<256; i++) {

            mdct_window[i] = enc_window[i]/(enc_window[i]*enc_window[i] + enc_window[255-i]*enc_window[255-i]);

            mdct_window[511-i] = mdct_window[i];

        }



    /* Initialize the MDCT transform. */

    ff_mdct_init(&mdct_ctx, 9, 1, 1.0);

}
