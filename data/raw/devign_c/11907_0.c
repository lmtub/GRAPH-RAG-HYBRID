CaptureVoiceOut *AUD_add_capture (

    struct audsettings *as,

    struct audio_capture_ops *ops,

    void *cb_opaque

    )

{

    AudioState *s = &glob_audio_state;

    CaptureVoiceOut *cap;

    struct capture_callback *cb;



    if (audio_validate_settings (as)) {

        dolog ("Invalid settings were passed when trying to add capture\n");

        audio_print_settings (as);

        goto err0;

    }



    cb = audio_calloc (AUDIO_FUNC, 1, sizeof (*cb));

    if (!cb) {

        dolog ("Could not allocate capture callback information, size %zu\n",

               sizeof (*cb));

        goto err0;

    }

    cb->ops = *ops;

    cb->opaque = cb_opaque;



    cap = audio_pcm_capture_find_specific (as);

    if (cap) {

        LIST_INSERT_HEAD (&cap->cb_head, cb, entries);

        return cap;

    }

    else {

        HWVoiceOut *hw;

        CaptureVoiceOut *cap;



        cap = audio_calloc (AUDIO_FUNC, 1, sizeof (*cap));

        if (!cap) {

            dolog ("Could not allocate capture voice, size %zu\n",

                   sizeof (*cap));

            goto err1;

        }



        hw = &cap->hw;

        LIST_INIT (&hw->sw_head);

        LIST_INIT (&cap->cb_head);



        /* XXX find a more elegant way */

        hw->samples = 4096 * 4;

        hw->mix_buf = audio_calloc (AUDIO_FUNC, hw->samples,

                                    sizeof (struct st_sample));

        if (!hw->mix_buf) {

            dolog ("Could not allocate capture mix buffer (%d samples)\n",

                   hw->samples);

            goto err2;

        }



        audio_pcm_init_info (&hw->info, as);



        cap->buf = audio_calloc (AUDIO_FUNC, hw->samples, 1 << hw->info.shift);

        if (!cap->buf) {

            dolog ("Could not allocate capture buffer "

                   "(%d samples, each %d bytes)\n",

                   hw->samples, 1 << hw->info.shift);

            goto err3;

        }



        hw->clip = mixeng_clip

            [hw->info.nchannels == 2]

            [hw->info.sign]

            [hw->info.swap_endianness]

            [audio_bits_to_index (hw->info.bits)];



        LIST_INSERT_HEAD (&s->cap_head, cap, entries);

        LIST_INSERT_HEAD (&cap->cb_head, cb, entries);



        hw = NULL;

        while ((hw = audio_pcm_hw_find_any_out (hw))) {

            audio_attach_capture (hw);

        }

        return cap;



    err3:

        qemu_free (cap->hw.mix_buf);

    err2:

        qemu_free (cap);

    err1:

        qemu_free (cb);

    err0:

        return NULL;

    }

}
