static int synchronize_audio(VideoState *is, short *samples,

                             int samples_size1, double pts)

{

    int n, samples_size;

    double ref_clock;



    n = 2 * is->audio_st->codec->channels;

    samples_size = samples_size1;



    /* if not master, then we try to remove or add samples to correct the clock */

    if (((is->av_sync_type == AV_SYNC_VIDEO_MASTER && is->video_st) ||

         is->av_sync_type == AV_SYNC_EXTERNAL_CLOCK)) {

        double diff, avg_diff;

        int wanted_size, min_size, max_size, nb_samples;



        ref_clock = get_master_clock(is);

        diff = get_audio_clock(is) - ref_clock;



        if (diff < AV_NOSYNC_THRESHOLD) {

            is->audio_diff_cum = diff + is->audio_diff_avg_coef * is->audio_diff_cum;

            if (is->audio_diff_avg_count < AUDIO_DIFF_AVG_NB) {

                /* not enough measures to have a correct estimate */

                is->audio_diff_avg_count++;

            } else {

                /* estimate the A-V difference */

                avg_diff = is->audio_diff_cum * (1.0 - is->audio_diff_avg_coef);



                if (fabs(avg_diff) >= is->audio_diff_threshold) {

                    wanted_size = samples_size + ((int)(diff * is->audio_st->codec->sample_rate) * n);

                    nb_samples = samples_size / n;



                    min_size = ((nb_samples * (100 - SAMPLE_CORRECTION_PERCENT_MAX)) / 100) * n;

                    max_size = ((nb_samples * (100 + SAMPLE_CORRECTION_PERCENT_MAX)) / 100) * n;

                    if (wanted_size < min_size)

                        wanted_size = min_size;

                    else if (wanted_size > max_size)

                        wanted_size = max_size;



                    /* add or remove samples to correction the synchro */

                    if (wanted_size < samples_size) {

                        /* remove samples */

                        samples_size = wanted_size;

                    } else if (wanted_size > samples_size) {

                        uint8_t *samples_end, *q;

                        int nb;



                        /* add samples */

                        nb = (samples_size - wanted_size);

                        samples_end = (uint8_t *)samples + samples_size - n;

                        q = samples_end + n;

                        while (nb > 0) {

                            memcpy(q, samples_end, n);

                            q += n;

                            nb -= n;

                        }

                        samples_size = wanted_size;

                    }

                }

                av_dlog(NULL, "diff=%f adiff=%f sample_diff=%d apts=%0.3f vpts=%0.3f %f\n",

                        diff, avg_diff, samples_size - samples_size1,

                        is->audio_clock, is->video_clock, is->audio_diff_threshold);

            }

        } else {

            /* too big difference : may be initial PTS errors, so

               reset A-V filter */

            is->audio_diff_avg_count = 0;

            is->audio_diff_cum       = 0;

        }

    }



    return samples_size;

}
