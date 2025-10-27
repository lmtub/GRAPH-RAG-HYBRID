void audio_encode_example(const char *filename)

{

    AVCodec *codec;

    AVCodecContext *c= NULL;

    int frame_size, i, j, out_size, outbuf_size;

    FILE *f;

    short *samples;

    float t, tincr;

    uint8_t *outbuf;



    printf("Audio encoding\n");



    /* find the MP2 encoder */

    codec = avcodec_find_encoder(CODEC_ID_MP2);

    if (!codec) {

        fprintf(stderr, "codec not found\n");

        exit(1);

    }



    c= avcodec_alloc_context();

    

    /* put sample parameters */

    c->bit_rate = 64000;

    c->sample_rate = 44100;

    c->channels = 2;



    /* open it */

    if (avcodec_open(c, codec) < 0) {

        fprintf(stderr, "could not open codec\n");

        exit(1);

    }

    

    /* the codec gives us the frame size, in samples */

    frame_size = c->frame_size;

    samples = malloc(frame_size * 2 * c->channels);

    outbuf_size = 10000;

    outbuf = malloc(outbuf_size);



    f = fopen(filename, "w");

    if (!f) {

        fprintf(stderr, "could not open %s\n", filename);

        exit(1);

    }

        

    /* encode a single tone sound */

    t = 0;

    tincr = 2 * M_PI * 440.0 / c->sample_rate;

    for(i=0;i<200;i++) {

        for(j=0;j<frame_size;j++) {

            samples[2*j] = (int)(sin(t) * 10000);

            samples[2*j+1] = samples[2*j];

            t += tincr;

        }

        /* encode the samples */

        out_size = avcodec_encode_audio(c, outbuf, outbuf_size, samples);

        fwrite(outbuf, 1, out_size, f);

    }

    fclose(f);

    free(outbuf);

    free(samples);



    avcodec_close(c);

    free(c);

}
