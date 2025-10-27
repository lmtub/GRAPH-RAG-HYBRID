static int no_init_out (HWVoiceOut *hw, audsettings_t *as)

{

    audio_pcm_init_info (&hw->info, as);

    hw->samples = 1024;

    return 0;

}
