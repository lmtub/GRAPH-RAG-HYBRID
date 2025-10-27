static int output_data_internal(MLPDecodeContext *m, unsigned int substr,

                                uint8_t *data, unsigned int *data_size, int is32)

{

    SubStream *s = &m->substream[substr];

    unsigned int i, out_ch = 0;

    int32_t *data_32 = (int32_t*) data;

    int16_t *data_16 = (int16_t*) data;



    if (*data_size < (s->max_channel + 1) * s->blockpos * (is32 ? 4 : 2))

        return -1;



    for (i = 0; i < s->blockpos; i++) {

        for (out_ch = 0; out_ch <= s->max_matrix_channel; out_ch++) {

            int mat_ch = s->ch_assign[out_ch];

            int32_t sample = m->sample_buffer[i][mat_ch]

                          << s->output_shift[mat_ch];

            s->lossless_check_data ^= (sample & 0xffffff) << mat_ch;

            if (is32) *data_32++ = sample << 8;

            else      *data_16++ = sample >> 8;

        }

    }



    *data_size = i * out_ch * (is32 ? 4 : 2);



    return 0;

}
