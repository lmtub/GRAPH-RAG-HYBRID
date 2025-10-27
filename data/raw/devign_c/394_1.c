static int wma_decode_block(WMACodecContext *s)

{

    int n, v, a, ch, bsize;

    int coef_nb_bits, total_gain;

    int nb_coefs[MAX_CHANNELS];

    float mdct_norm;

    FFTContext *mdct;



#ifdef TRACE

    tprintf(s->avctx, "***decode_block: %d:%d\n", s->frame_count - 1, s->block_num);

#endif



    /* compute current block length */

    if (s->use_variable_block_len) {

        n = av_log2(s->nb_block_sizes - 1) + 1;



        if (s->reset_block_lengths) {

            s->reset_block_lengths = 0;

            v = get_bits(&s->gb, n);

            if (v >= s->nb_block_sizes){

                av_log(s->avctx, AV_LOG_ERROR, "prev_block_len_bits %d out of range\n", s->frame_len_bits - v);

                return -1;


            s->prev_block_len_bits = s->frame_len_bits - v;

            v = get_bits(&s->gb, n);

            if (v >= s->nb_block_sizes){

                av_log(s->avctx, AV_LOG_ERROR, "block_len_bits %d out of range\n", s->frame_len_bits - v);

                return -1;


            s->block_len_bits = s->frame_len_bits - v;

        } else {

            /* update block lengths */

            s->prev_block_len_bits = s->block_len_bits;

            s->block_len_bits = s->next_block_len_bits;


        v = get_bits(&s->gb, n);

        if (v >= s->nb_block_sizes){

            av_log(s->avctx, AV_LOG_ERROR, "next_block_len_bits %d out of range\n", s->frame_len_bits - v);

            return -1;


        s->next_block_len_bits = s->frame_len_bits - v;

    } else {

        /* fixed block len */

        s->next_block_len_bits = s->frame_len_bits;

        s->prev_block_len_bits = s->frame_len_bits;

        s->block_len_bits = s->frame_len_bits;




    if (s->frame_len_bits - s->block_len_bits >= s->nb_block_sizes){

        av_log(s->avctx, AV_LOG_ERROR, "block_len_bits not initialized to a valid value\n");

        return -1;




    /* now check if the block length is coherent with the frame length */

    s->block_len = 1 << s->block_len_bits;

    if ((s->block_pos + s->block_len) > s->frame_len){

        av_log(s->avctx, AV_LOG_ERROR, "frame_len overflow\n");

        return -1;




    if (s->avctx->channels == 2) {

        s->ms_stereo = get_bits1(&s->gb);


    v = 0;

    for(ch = 0; ch < s->avctx->channels; ch++) {

        a = get_bits1(&s->gb);

        s->channel_coded[ch] = a;

        v |= a;




    bsize = s->frame_len_bits - s->block_len_bits;



    /* if no channel coded, no need to go further */

    /* XXX: fix potential framing problems */

    if (!v)

        goto next;



    /* read total gain and extract corresponding number of bits for

       coef escape coding */

    total_gain = 1;

    for(;;) {





        a = get_bits(&s->gb, 7);

        total_gain += a;

        if (a != 127)

            break;




    coef_nb_bits= ff_wma_total_gain_to_bits(total_gain);



    /* compute number of coefficients */

    n = s->coefs_end[bsize] - s->coefs_start;

    for(ch = 0; ch < s->avctx->channels; ch++)

        nb_coefs[ch] = n;



    /* complex coding */

    if (s->use_noise_coding) {



        for(ch = 0; ch < s->avctx->channels; ch++) {

            if (s->channel_coded[ch]) {

                int i, n, a;

                n = s->exponent_high_sizes[bsize];

                for(i=0;i<n;i++) {

                    a = get_bits1(&s->gb);

                    s->high_band_coded[ch][i] = a;

                    /* if noise coding, the coefficients are not transmitted */

                    if (a)

                        nb_coefs[ch] -= s->exponent_high_bands[bsize][i];




        for(ch = 0; ch < s->avctx->channels; ch++) {

            if (s->channel_coded[ch]) {

                int i, n, val, code;



                n = s->exponent_high_sizes[bsize];

                val = (int)0x80000000;

                for(i=0;i<n;i++) {

                    if (s->high_band_coded[ch][i]) {

                        if (val == (int)0x80000000) {

                            val = get_bits(&s->gb, 7) - 19;

                        } else {

                            code = get_vlc2(&s->gb, s->hgain_vlc.table, HGAINVLCBITS, HGAINMAX);

                            if (code < 0){

                                av_log(s->avctx, AV_LOG_ERROR, "hgain vlc invalid\n");

                                return -1;


                            val += code - 18;


                        s->high_band_values[ch][i] = val;








    /* exponents can be reused in short blocks. */

    if ((s->block_len_bits == s->frame_len_bits) ||

        get_bits1(&s->gb)) {

        for(ch = 0; ch < s->avctx->channels; ch++) {

            if (s->channel_coded[ch]) {

                if (s->use_exp_vlc) {

                    if (decode_exp_vlc(s, ch) < 0)

                        return -1;

                } else {

                    decode_exp_lsp(s, ch);


                s->exponents_bsize[ch] = bsize;






    /* parse spectral coefficients : just RLE encoding */

    for (ch = 0; ch < s->avctx->channels; ch++) {

        if (s->channel_coded[ch]) {

            int tindex;

            WMACoef* ptr = &s->coefs1[ch][0];



            /* special VLC tables are used for ms stereo because

               there is potentially less energy there */

            tindex = (ch == 1 && s->ms_stereo);

            memset(ptr, 0, s->block_len * sizeof(WMACoef));

            ff_wma_run_level_decode(s->avctx, &s->gb, &s->coef_vlc[tindex],

                  s->level_table[tindex], s->run_table[tindex],

                  0, ptr, 0, nb_coefs[ch],

                  s->block_len, s->frame_len_bits, coef_nb_bits);


        if (s->version == 1 && s->avctx->channels >= 2) {

            align_get_bits(&s->gb);





    /* normalize */

    {

        int n4 = s->block_len / 2;

        mdct_norm = 1.0 / (float)n4;

        if (s->version == 1) {

            mdct_norm *= sqrt(n4);





    /* finally compute the MDCT coefficients */

    for (ch = 0; ch < s->avctx->channels; ch++) {

        if (s->channel_coded[ch]) {

            WMACoef *coefs1;

            float *coefs, *exponents, mult, mult1, noise;

            int i, j, n, n1, last_high_band, esize;

            float exp_power[HIGH_BAND_MAX_SIZE];



            coefs1 = s->coefs1[ch];

            exponents = s->exponents[ch];

            esize = s->exponents_bsize[ch];

            mult = pow(10, total_gain * 0.05) / s->max_exponent[ch];

            mult *= mdct_norm;

            coefs = s->coefs[ch];

            if (s->use_noise_coding) {

                mult1 = mult;

                /* very low freqs : noise */

                for(i = 0;i < s->coefs_start; i++) {

                    *coefs++ = s->noise_table[s->noise_index] *

                      exponents[i<<bsize>>esize] * mult1;

                    s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);




                n1 = s->exponent_high_sizes[bsize];



                /* compute power of high bands */

                exponents = s->exponents[ch] +

                    (s->high_band_start[bsize]<<bsize>>esize);

                last_high_band = 0; /* avoid warning */

                for(j=0;j<n1;j++) {

                    n = s->exponent_high_bands[s->frame_len_bits -

                                              s->block_len_bits][j];

                    if (s->high_band_coded[ch][j]) {

                        float e2, v;

                        e2 = 0;

                        for(i = 0;i < n; i++) {

                            v = exponents[i<<bsize>>esize];

                            e2 += v * v;


                        exp_power[j] = e2 / n;

                        last_high_band = j;

                        tprintf(s->avctx, "%d: power=%f (%d)\n", j, exp_power[j], n);


                    exponents += n<<bsize>>esize;




                /* main freqs and high freqs */

                exponents = s->exponents[ch] + (s->coefs_start<<bsize>>esize);

                for(j=-1;j<n1;j++) {

                    if (j < 0) {

                        n = s->high_band_start[bsize] -

                            s->coefs_start;

                    } else {

                        n = s->exponent_high_bands[s->frame_len_bits -

                                                  s->block_len_bits][j];


                    if (j >= 0 && s->high_band_coded[ch][j]) {

                        /* use noise with specified power */

                        mult1 = sqrt(exp_power[j] / exp_power[last_high_band]);

                        /* XXX: use a table */

                        mult1 = mult1 * pow(10, s->high_band_values[ch][j] * 0.05);

                        mult1 = mult1 / (s->max_exponent[ch] * s->noise_mult);

                        mult1 *= mdct_norm;

                        for(i = 0;i < n; i++) {

                            noise = s->noise_table[s->noise_index];

                            s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);

                            *coefs++ =  noise *

                                exponents[i<<bsize>>esize] * mult1;


                        exponents += n<<bsize>>esize;

                    } else {

                        /* coded values + small noise */

                        for(i = 0;i < n; i++) {

                            noise = s->noise_table[s->noise_index];

                            s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);

                            *coefs++ = ((*coefs1++) + noise) *

                                exponents[i<<bsize>>esize] * mult;


                        exponents += n<<bsize>>esize;





                /* very high freqs : noise */

                n = s->block_len - s->coefs_end[bsize];

                mult1 = mult * exponents[((-1<<bsize))>>esize];

                for(i = 0; i < n; i++) {

                    *coefs++ = s->noise_table[s->noise_index] * mult1;

                    s->noise_index = (s->noise_index + 1) & (NOISE_TAB_SIZE - 1);


            } else {

                /* XXX: optimize more */

                for(i = 0;i < s->coefs_start; i++)

                    *coefs++ = 0.0;

                n = nb_coefs[ch];

                for(i = 0;i < n; i++) {

                    *coefs++ = coefs1[i] * exponents[i<<bsize>>esize] * mult;


                n = s->block_len - s->coefs_end[bsize];

                for(i = 0;i < n; i++)

                    *coefs++ = 0.0;






#ifdef TRACE

    for (ch = 0; ch < s->avctx->channels; ch++) {

        if (s->channel_coded[ch]) {

            dump_floats(s, "exponents", 3, s->exponents[ch], s->block_len);

            dump_floats(s, "coefs", 1, s->coefs[ch], s->block_len);



#endif



    if (s->ms_stereo && s->channel_coded[1]) {

        /* nominal case for ms stereo: we do it before mdct */

        /* no need to optimize this case because it should almost

           never happen */

        if (!s->channel_coded[0]) {

            tprintf(s->avctx, "rare ms-stereo case happened\n");

            memset(s->coefs[0], 0, sizeof(float) * s->block_len);

            s->channel_coded[0] = 1;




        s->fdsp.butterflies_float(s->coefs[0], s->coefs[1], s->block_len);




next:

    mdct = &s->mdct_ctx[bsize];



    for (ch = 0; ch < s->avctx->channels; ch++) {

        int n4, index;



        n4 = s->block_len / 2;

        if(s->channel_coded[ch]){

            mdct->imdct_calc(mdct, s->output, s->coefs[ch]);

        }else if(!(s->ms_stereo && ch==1))

            memset(s->output, 0, sizeof(s->output));



        /* multiply by the window and add in the frame */

        index = (s->frame_len / 2) + s->block_pos - n4;

        wma_window(s, &s->frame_out[ch][index]);




    /* update block number */

    s->block_num++;

    s->block_pos += s->block_len;

    if (s->block_pos >= s->frame_len)

        return 1;

    else

        return 0;
