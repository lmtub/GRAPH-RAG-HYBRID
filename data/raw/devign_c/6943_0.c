static int sonic_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                              const AVFrame *frame, int *got_packet_ptr)

{

    SonicContext *s = avctx->priv_data;

    RangeCoder c;

    int i, j, ch, quant = 0, x = 0;

    int ret;

    const short *samples = (const int16_t*)frame->data[0];

    uint8_t state[32];



    if ((ret = ff_alloc_packet2(avctx, avpkt, s->frame_size * 5 + 1000)) < 0)

        return ret;



    ff_init_range_encoder(&c, avpkt->data, avpkt->size);

    ff_build_rac_states(&c, 0.05*(1LL<<32), 256-8);

    memset(state, 128, sizeof(state));



    // short -> internal

    for (i = 0; i < s->frame_size; i++)

        s->int_samples[i] = samples[i];



    if (!s->lossless)

        for (i = 0; i < s->frame_size; i++)

            s->int_samples[i] = s->int_samples[i] << SAMPLE_SHIFT;



    switch(s->decorrelation)

    {

        case MID_SIDE:

            for (i = 0; i < s->frame_size; i += s->channels)

            {

                s->int_samples[i] += s->int_samples[i+1];

                s->int_samples[i+1] -= shift(s->int_samples[i], 1);

            }

            break;

        case LEFT_SIDE:

            for (i = 0; i < s->frame_size; i += s->channels)

                s->int_samples[i+1] -= s->int_samples[i];

            break;

        case RIGHT_SIDE:

            for (i = 0; i < s->frame_size; i += s->channels)

                s->int_samples[i] -= s->int_samples[i+1];

            break;

    }



    memset(s->window, 0, 4* s->window_size);



    for (i = 0; i < s->tail_size; i++)

        s->window[x++] = s->tail[i];



    for (i = 0; i < s->frame_size; i++)

        s->window[x++] = s->int_samples[i];



    for (i = 0; i < s->tail_size; i++)

        s->window[x++] = 0;



    for (i = 0; i < s->tail_size; i++)

        s->tail[i] = s->int_samples[s->frame_size - s->tail_size + i];



    // generate taps

    modified_levinson_durbin(s->window, s->window_size,

                s->predictor_k, s->num_taps, s->channels, s->tap_quant);

    if ((ret = intlist_write(&c, state, s->predictor_k, s->num_taps, 0)) < 0)

        return ret;



    for (ch = 0; ch < s->channels; ch++)

    {

        x = s->tail_size+ch;

        for (i = 0; i < s->block_align; i++)

        {

            int sum = 0;

            for (j = 0; j < s->downsampling; j++, x += s->channels)

                sum += s->window[x];

            s->coded_samples[ch][i] = sum;

        }

    }



    // simple rate control code

    if (!s->lossless)

    {

        double energy1 = 0.0, energy2 = 0.0;

        for (ch = 0; ch < s->channels; ch++)

        {

            for (i = 0; i < s->block_align; i++)

            {

                double sample = s->coded_samples[ch][i];

                energy2 += sample*sample;

                energy1 += fabs(sample);

            }

        }



        energy2 = sqrt(energy2/(s->channels*s->block_align));

        energy1 = M_SQRT2*energy1/(s->channels*s->block_align);



        // increase bitrate when samples are like a gaussian distribution

        // reduce bitrate when samples are like a two-tailed exponential distribution



        if (energy2 > energy1)

            energy2 += (energy2-energy1)*RATE_VARIATION;



        quant = (int)(BASE_QUANT*s->quantization*energy2/SAMPLE_FACTOR);

//        av_log(avctx, AV_LOG_DEBUG, "quant: %d energy: %f / %f\n", quant, energy1, energy2);



        quant = av_clip(quant, 1, 65534);



        put_symbol(&c, state, quant, 0, NULL, NULL);



        quant *= SAMPLE_FACTOR;

    }



    // write out coded samples

    for (ch = 0; ch < s->channels; ch++)

    {

        if (!s->lossless)

            for (i = 0; i < s->block_align; i++)

                s->coded_samples[ch][i] = ROUNDED_DIV(s->coded_samples[ch][i], quant);



        if ((ret = intlist_write(&c, state, s->coded_samples[ch], s->block_align, 1)) < 0)

            return ret;

    }



//    av_log(avctx, AV_LOG_DEBUG, "used bytes: %d\n", (put_bits_count(&pb)+7)/8);



    avpkt->size = ff_rac_terminate(&c);

    *got_packet_ptr = 1;

    return 0;



}
