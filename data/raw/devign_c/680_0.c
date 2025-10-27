static inline void do_imdct(AC3DecodeContext *s, int channels)

{

    int ch;



    for (ch=1; ch<=channels; ch++) {

        if (s->block_switch[ch]) {

            int i;

            float *x = s->tmp_output+128;

            for(i=0; i<128; i++)

                x[i] = s->transform_coeffs[ch][2*i];

            ff_imdct_half(&s->imdct_256, s->tmp_output, x);

            s->dsp.vector_fmul_window(s->output[ch-1], s->delay[ch-1], s->tmp_output, s->window, s->add_bias, 128);

            for(i=0; i<128; i++)

                x[i] = s->transform_coeffs[ch][2*i+1];

            ff_imdct_half(&s->imdct_256, s->delay[ch-1], x);

        } else {

            ff_imdct_half(&s->imdct_512, s->tmp_output, s->transform_coeffs[ch]);

            s->dsp.vector_fmul_window(s->output[ch-1], s->delay[ch-1], s->tmp_output, s->window, s->add_bias, 128);

            memcpy(s->delay[ch-1], s->tmp_output+128, 128*sizeof(float));

        }

    }

}
