static void sdl_audio_callback(void *opaque, Uint8 *stream, int len)

{

    VideoState *is = opaque;

    int audio_size, len1, silence = 0;



    audio_callback_time = av_gettime_relative();



    while (len > 0) {

        if (is->audio_buf_index >= is->audio_buf_size) {

           audio_size = audio_decode_frame(is);

           if (audio_size < 0) {

                /* if error, just output silence */

               silence = 1;

               is->audio_buf_size = SDL_AUDIO_MIN_BUFFER_SIZE / is->audio_tgt.frame_size * is->audio_tgt.frame_size;

           } else {

               if (is->show_mode != SHOW_MODE_VIDEO)

                   update_sample_display(is, (int16_t *)is->audio_buf, audio_size);

               is->audio_buf_size = audio_size;

           }

           is->audio_buf_index = 0;

        }

        len1 = is->audio_buf_size - is->audio_buf_index;

        if (len1 > len)

            len1 = len;

        if (!is->muted && !silence && is->audio_volume == SDL_MIX_MAXVOLUME)

            memcpy(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1);

        else {

            memset(stream, 0, len1);

            if (!is->muted && !silence)

                SDL_MixAudio(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1, is->audio_volume);

        }

        len -= len1;

        stream += len1;

        is->audio_buf_index += len1;

    }

    is->audio_write_buf_size = is->audio_buf_size - is->audio_buf_index;

    /* Let's assume the audio driver that is used by SDL has two periods. */

    if (!isnan(is->audio_clock)) {

        set_clock_at(&is->audclk, is->audio_clock - (double)(2 * is->audio_hw_buf_size + is->audio_write_buf_size) / is->audio_tgt.bytes_per_sec, is->audio_clock_serial, audio_callback_time / 1000000.0);

        sync_clock_to_slave(&is->extclk, &is->audclk);

    }

}
