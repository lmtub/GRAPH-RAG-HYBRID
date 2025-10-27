static void qdm2_calculate_fft (QDM2Context *q, int channel, int sub_packet)

{

    const float gain = (q->channels == 1 && q->nb_channels == 2) ? 0.5f : 1.0f;

    int i;

    q->fft.complex[channel][0].re *= 2.0f;

    q->fft.complex[channel][0].im = 0.0f;

    q->rdft_ctx.rdft_calc(&q->rdft_ctx, (FFTSample *)q->fft.complex[channel]);

    /* add samples to output buffer */

    for (i = 0; i < ((q->fft_frame_size + 15) & ~15); i++)

        q->output_buffer[q->channels * i + channel] += ((float *) q->fft.complex[channel])[i] * gain;

}
