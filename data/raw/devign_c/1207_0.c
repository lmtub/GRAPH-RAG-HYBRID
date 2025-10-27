int main(int argc, char **argv){

    int in_sample_rate, out_sample_rate, ch ,i, flush_count;

    uint64_t in_ch_layout, out_ch_layout;

    enum AVSampleFormat in_sample_fmt, out_sample_fmt;

    uint8_t array_in[SAMPLES*8*8];

    uint8_t array_mid[SAMPLES*8*8*3];

    uint8_t array_out[SAMPLES*8*8+100];

    uint8_t *ain[SWR_CH_MAX];

    uint8_t *aout[SWR_CH_MAX];

    uint8_t *amid[SWR_CH_MAX];

    int flush_i=0;

    int mode;

    int num_tests = 10000;

    uint32_t seed = 0;

    uint32_t rand_seed = 0;

    int remaining_tests[FF_ARRAY_ELEMS(rates) * FF_ARRAY_ELEMS(layouts) * FF_ARRAY_ELEMS(formats) * FF_ARRAY_ELEMS(layouts) * FF_ARRAY_ELEMS(formats)];

    int max_tests = FF_ARRAY_ELEMS(remaining_tests);

    int test;

    int specific_test= -1;



    struct SwrContext * forw_ctx= NULL;

    struct SwrContext *backw_ctx= NULL;



    if (argc > 1) {

        if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {

            av_log(NULL, AV_LOG_INFO, "Usage: swresample-test [<num_tests>[ <test>]]  \n"

                   "num_tests           Default is %d\n", num_tests);

            return 0;

        }

        num_tests = strtol(argv[1], NULL, 0);

        if(num_tests < 0) {

            num_tests = -num_tests;

            rand_seed = time(0);

        }

        if(num_tests<= 0 || num_tests>max_tests)

            num_tests = max_tests;

        if(argc > 2) {

            specific_test = strtol(argv[1], NULL, 0);

        }

    }



    for(i=0; i<max_tests; i++)

        remaining_tests[i] = i;



    for(test=0; test<num_tests; test++){

        unsigned r;

        uint_rand(seed);

        r = (seed * (uint64_t)(max_tests - test)) >>32;

        FFSWAP(int, remaining_tests[r], remaining_tests[max_tests - test - 1]);

    }

    qsort(remaining_tests + max_tests - num_tests, num_tests, sizeof(remaining_tests[0]), (void*)cmp);

    in_sample_rate=16000;

    for(test=0; test<num_tests; test++){

        char  in_layout_string[256];

        char out_layout_string[256];

        unsigned vector= remaining_tests[max_tests - test - 1];

        int in_ch_count;

        int out_count, mid_count, out_ch_count;



        in_ch_layout    = layouts[vector % FF_ARRAY_ELEMS(layouts)]; vector /= FF_ARRAY_ELEMS(layouts);

        out_ch_layout   = layouts[vector % FF_ARRAY_ELEMS(layouts)]; vector /= FF_ARRAY_ELEMS(layouts);

        in_sample_fmt   = formats[vector % FF_ARRAY_ELEMS(formats)]; vector /= FF_ARRAY_ELEMS(formats);

        out_sample_fmt  = formats[vector % FF_ARRAY_ELEMS(formats)]; vector /= FF_ARRAY_ELEMS(formats);

        out_sample_rate = rates  [vector % FF_ARRAY_ELEMS(rates  )]; vector /= FF_ARRAY_ELEMS(rates);

        av_assert0(!vector);



        if(specific_test == 0){

            if(out_sample_rate != in_sample_rate || in_ch_layout != out_ch_layout)

                continue;

        }



        in_ch_count= av_get_channel_layout_nb_channels(in_ch_layout);

        out_ch_count= av_get_channel_layout_nb_channels(out_ch_layout);

        av_get_channel_layout_string( in_layout_string, sizeof( in_layout_string),  in_ch_count,  in_ch_layout);

        av_get_channel_layout_string(out_layout_string, sizeof(out_layout_string), out_ch_count, out_ch_layout);

        fprintf(stderr, "TEST: %s->%s, rate:%5d->%5d, fmt:%s->%s\n",

                in_layout_string, out_layout_string,

                in_sample_rate, out_sample_rate,

                av_get_sample_fmt_name(in_sample_fmt), av_get_sample_fmt_name(out_sample_fmt));

        forw_ctx  = swr_alloc_set_opts(forw_ctx, out_ch_layout, out_sample_fmt,  out_sample_rate,

                                                    in_ch_layout,  in_sample_fmt,  in_sample_rate,

                                        0, 0);

        backw_ctx = swr_alloc_set_opts(backw_ctx, in_ch_layout,  in_sample_fmt,             in_sample_rate,

                                                    out_ch_layout, out_sample_fmt, out_sample_rate,

                                        0, 0);

        if(swr_init( forw_ctx) < 0)

            fprintf(stderr, "swr_init(->) failed\n");

        if(swr_init(backw_ctx) < 0)

            fprintf(stderr, "swr_init(<-) failed\n");

        if(!forw_ctx)

            fprintf(stderr, "Failed to init forw_cts\n");

        if(!backw_ctx)

            fprintf(stderr, "Failed to init backw_ctx\n");

                //FIXME test planar

        setup_array(ain , array_in ,  in_sample_fmt,   SAMPLES);

        setup_array(amid, array_mid, out_sample_fmt, 3*SAMPLES);

        setup_array(aout, array_out,  in_sample_fmt           ,   SAMPLES);

#if 0

        for(ch=0; ch<in_ch_count; ch++){

            for(i=0; i<SAMPLES; i++)

                set(ain, ch, i, in_ch_count, in_sample_fmt, sin(i*i*3/SAMPLES));

        }

#else

        audiogen(ain, in_sample_fmt, in_ch_count, SAMPLES/6+1, SAMPLES);

#endif

        mode = uint_rand(rand_seed) % 3;

        if(mode==0 /*|| out_sample_rate == in_sample_rate*/) {

            mid_count= swr_convert(forw_ctx, amid, 3*SAMPLES, (const uint8_t **)ain, SAMPLES);

        } else if(mode==1){

            mid_count= swr_convert(forw_ctx, amid,         0, (const uint8_t **)ain, SAMPLES);

            mid_count+=swr_convert(forw_ctx, amid, 3*SAMPLES, (const uint8_t **)ain,       0);

        } else {

            int tmp_count;

            mid_count= swr_convert(forw_ctx, amid,         0, (const uint8_t **)ain,       1);

            av_assert0(mid_count==0);

            shift(ain,  1, in_ch_count, in_sample_fmt);

            mid_count+=swr_convert(forw_ctx, amid, 3*SAMPLES, (const uint8_t **)ain,       0);

            shift(amid,  mid_count, out_ch_count, out_sample_fmt); tmp_count = mid_count;

            mid_count+=swr_convert(forw_ctx, amid,         2, (const uint8_t **)ain,       2);

            shift(amid,  mid_count-tmp_count, out_ch_count, out_sample_fmt); tmp_count = mid_count;

            shift(ain,  2, in_ch_count, in_sample_fmt);

            mid_count+=swr_convert(forw_ctx, amid,         1, (const uint8_t **)ain, SAMPLES-3);

            shift(amid,  mid_count-tmp_count, out_ch_count, out_sample_fmt); tmp_count = mid_count;

            shift(ain, -3, in_ch_count, in_sample_fmt);

            mid_count+=swr_convert(forw_ctx, amid, 3*SAMPLES, (const uint8_t **)ain,       0);

            shift(amid,  -tmp_count, out_ch_count, out_sample_fmt);

        }

        out_count= swr_convert(backw_ctx,aout, SAMPLES, (const uint8_t **)amid, mid_count);



        for(ch=0; ch<in_ch_count; ch++){

            double sse, maxdiff=0;

            double sum_a= 0;

            double sum_b= 0;

            double sum_aa= 0;

            double sum_bb= 0;

            double sum_ab= 0;

            for(i=0; i<out_count; i++){

                double a= get(ain , ch, i, in_ch_count, in_sample_fmt);

                double b= get(aout, ch, i, in_ch_count, in_sample_fmt);

                sum_a += a;

                sum_b += b;

                sum_aa+= a*a;

                sum_bb+= b*b;

                sum_ab+= a*b;

                maxdiff= FFMAX(maxdiff, FFABS(a-b));

            }

            sse= sum_aa + sum_bb - 2*sum_ab;

            if(sse < 0 && sse > -0.00001) sse=0; //fix rounding error



            fprintf(stderr, "[e:%f c:%f max:%f] len:%5d\n", sqrt(sse/out_count), sum_ab/(sqrt(sum_aa*sum_bb)), maxdiff, out_count);

        }



        flush_i++;

        flush_i%=21;

        flush_count = swr_convert(backw_ctx,aout, flush_i, 0, 0);

        shift(aout,  flush_i, in_ch_count, in_sample_fmt);

        flush_count+= swr_convert(backw_ctx,aout, SAMPLES-flush_i, 0, 0);

        shift(aout, -flush_i, in_ch_count, in_sample_fmt);

        if(flush_count){

            for(ch=0; ch<in_ch_count; ch++){

                double sse, maxdiff=0;

                double sum_a= 0;

                double sum_b= 0;

                double sum_aa= 0;

                double sum_bb= 0;

                double sum_ab= 0;

                for(i=0; i<flush_count; i++){

                    double a= get(ain , ch, i+out_count, in_ch_count, in_sample_fmt);

                    double b= get(aout, ch, i, in_ch_count, in_sample_fmt);

                    sum_a += a;

                    sum_b += b;

                    sum_aa+= a*a;

                    sum_bb+= b*b;

                    sum_ab+= a*b;

                    maxdiff= FFMAX(maxdiff, FFABS(a-b));

                }

                sse= sum_aa + sum_bb - 2*sum_ab;

                if(sse < 0 && sse > -0.00001) sse=0; //fix rounding error



                fprintf(stderr, "[e:%f c:%f max:%f] len:%5d F:%3d\n", sqrt(sse/flush_count), sum_ab/(sqrt(sum_aa*sum_bb)), maxdiff, flush_count, flush_i);

            }

        }





        fprintf(stderr, "\n");

    }



    return 0;

}
