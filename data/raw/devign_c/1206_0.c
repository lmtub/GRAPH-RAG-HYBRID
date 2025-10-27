mlt_compensate_output(COOKContext *q, float *decode_buffer,

                      cook_gains *gains, float *previous_buffer,

                      int16_t *out, int chan)

{

    int j;



    cook_imlt(q, decode_buffer, q->mono_mdct_output);

    gain_compensate(q, gains, previous_buffer);



    /* Clip and convert floats to 16 bits.

     */

    for (j = 0; j < q->samples_per_channel; j++) {

        out[chan + q->nb_channels * j] =

          av_clip(lrintf(q->mono_mdct_output[j]), -32768, 32767);

    }

}
