static void qdm2_init(QDM2Context *q) {

    static int inited = 0;



    if (inited != 0)

        return;

    inited = 1;



    qdm2_init_vlc();

    ff_mpa_synth_init(mpa_window);

    softclip_table_init();

    rnd_table_init();

    init_noise_samples();



    av_log(NULL, AV_LOG_DEBUG, "init done\n");

}
