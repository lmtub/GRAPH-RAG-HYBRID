static void qpa_fini_out (HWVoiceOut *hw)

{

    void *ret;

    PAVoiceOut *pa = (PAVoiceOut *) hw;



    audio_pt_lock (&pa->pt, AUDIO_FUNC);

    pa->done = 1;

    audio_pt_unlock_and_signal (&pa->pt, AUDIO_FUNC);

    audio_pt_join (&pa->pt, &ret, AUDIO_FUNC);



    if (pa->s) {

        pa_simple_free (pa->s);

        pa->s = NULL;

    }



    audio_pt_fini (&pa->pt, AUDIO_FUNC);

    g_free (pa->pcm_buf);

    pa->pcm_buf = NULL;

}
