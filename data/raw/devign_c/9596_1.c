static void generate_2_noise_channels(MLPDecodeContext *m, unsigned int substr)

{

    SubStream *s = &m->substream[substr];

    unsigned int i;

    uint32_t seed = s->noisegen_seed;

    unsigned int maxchan = s->max_matrix_channel;



    for (i = 0; i < s->blockpos; i++) {

        uint16_t seed_shr7 = seed >> 7;

        m->sample_buffer[i][maxchan+1] = ((int8_t)(seed >> 15)) << s->noise_shift;

        m->sample_buffer[i][maxchan+2] = ((int8_t) seed_shr7)   << s->noise_shift;



        seed = (seed << 16) ^ seed_shr7 ^ (seed_shr7 << 5);

    }



    s->noisegen_seed = seed;

}
