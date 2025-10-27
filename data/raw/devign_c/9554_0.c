static void *qpa_thread_in (void *arg)

{

    PAVoiceIn *pa = arg;

    HWVoiceIn *hw = &pa->hw;

    int threshold;



    threshold = conf.divisor ? hw->samples / conf.divisor : 0;



    if (audio_pt_lock (&pa->pt, AUDIO_FUNC)) {

        return NULL;

    }



    for (;;) {

        int incr, to_grab, wpos;



        for (;;) {

            if (pa->done) {

                goto exit;

            }



            if (pa->dead > threshold) {

                break;

            }



            if (audio_pt_wait (&pa->pt, AUDIO_FUNC)) {

                goto exit;

            }

        }



        incr = to_grab = pa->dead;

        wpos = hw->wpos;



        if (audio_pt_unlock (&pa->pt, AUDIO_FUNC)) {

            return NULL;

        }



        while (to_grab) {

            int error;

            int chunk = audio_MIN (to_grab, hw->samples - wpos);

            void *buf = advance (pa->pcm_buf, wpos);



            if (pa_simple_read (pa->s, buf,

                                chunk << hw->info.shift, &error) < 0) {

                qpa_logerr (error, "pa_simple_read failed\n");

                return NULL;

            }



            hw->conv (hw->conv_buf + wpos, buf, chunk);

            wpos = (wpos + chunk) % hw->samples;

            to_grab -= chunk;

        }



        if (audio_pt_lock (&pa->pt, AUDIO_FUNC)) {

            return NULL;

        }



        pa->wpos = wpos;

        pa->dead -= incr;

        pa->incr += incr;

    }



 exit:

    audio_pt_unlock (&pa->pt, AUDIO_FUNC);

    return NULL;

}
