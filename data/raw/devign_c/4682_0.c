static void audio_encode_example(const char *filename)

{

    AVCodec *codec;

    AVCodecContext *c= NULL;

    AVFrame *frame;

    AVPacket pkt;

    int i, j, k, ret, got_output;

    int buffer_size;

    FILE *f;

    uint16_t *samples;

    float t, tincr;



    printf("Encode audio file %s\n", filename);



    /* find the MP2 encoder */

    codec = avcodec_find_encoder(AV_CODEC_ID_MP2);

    if (!codec) {

        fprintf(stderr, "Codec not found\n");

        exit(1);

    }



    c = avcodec_alloc_context3(codec);

    if (!c) {

        fprintf(stderr, "Could not allocate audio codec context\n");

        exit(1);

    }



    /* put sample parameters */

    c->bit_rate = 64000;



    /* check that the encoder supports s16 pcm input */

    c->sample_fmt = AV_SAMPLE_FMT_S16;

    if (!check_sample_fmt(codec, c->sample_fmt)) {

        fprintf(stderr, "Encoder does not support sample format %s",

                av_get_sample_fmt_name(c->sample_fmt));

        exit(1);

    }



    /* select other audio parameters supported by the encoder */

    c->sample_rate    = select_sample_rate(codec);

    c->channel_layout = select_channel_layout(codec);

    c->channels       = av_get_channel_layout_nb_channels(c->channel_layout);



    /* open it */

    if (avcodec_open2(c, codec, NULL) < 0) {

        fprintf(stderr, "Could not open codec\n");

        exit(1);

    }



    f = fopen(filename, "wb");

    if (!f) {

        fprintf(stderr, "Could not open %s\n", filename);

        exit(1);

    }



    /* frame containing input raw audio */

    frame = av_frame_alloc();

    if (!frame) {

        fprintf(stderr, "Could not allocate audio frame\n");

        exit(1);

    }



    frame->nb_samples     = c->frame_size;

    frame->format         = c->sample_fmt;

    frame->channel_layout = c->channel_layout;



    /* the codec gives us the frame size, in samples,

     * we calculate the size of the samples buffer in bytes */

    buffer_size = av_samples_get_buffer_size(NULL, c->channels, c->frame_size,

                                             c->sample_fmt, 0);

    if (!buffer_size) {

        fprintf(stderr, "Could not get sample buffer size\n");

        exit(1);

    }

    samples = av_malloc(buffer_size);

    if (!samples) {

        fprintf(stderr, "Could not allocate %d bytes for samples buffer\n",

                buffer_size);

        exit(1);

    }

    /* setup the data pointers in the AVFrame */

    ret = avcodec_fill_audio_frame(frame, c->channels, c->sample_fmt,

                                   (const uint8_t*)samples, buffer_size, 0);

    if (ret < 0) {

        fprintf(stderr, "Could not setup audio frame\n");

        exit(1);

    }



    /* encode a single tone sound */

    t = 0;

    tincr = 2 * M_PI * 440.0 / c->sample_rate;

    for(i=0;i<200;i++) {

        av_init_packet(&pkt);

        pkt.data = NULL; // packet data will be allocated by the encoder

        pkt.size = 0;



        for (j = 0; j < c->frame_size; j++) {

            samples[2*j] = (int)(sin(t) * 10000);



            for (k = 1; k < c->channels; k++)

                samples[2*j + k] = samples[2*j];

            t += tincr;

        }

        /* encode the samples */

        ret = avcodec_encode_audio2(c, &pkt, frame, &got_output);

        if (ret < 0) {

            fprintf(stderr, "Error encoding audio frame\n");

            exit(1);

        }

        if (got_output) {

            fwrite(pkt.data, 1, pkt.size, f);

            av_free_packet(&pkt);

        }

    }



    /* get the delayed frames */

    for (got_output = 1; got_output; i++) {

        ret = avcodec_encode_audio2(c, &pkt, NULL, &got_output);

        if (ret < 0) {

            fprintf(stderr, "Error encoding frame\n");

            exit(1);

        }



        if (got_output) {

            fwrite(pkt.data, 1, pkt.size, f);

            av_free_packet(&pkt);

        }

    }

    fclose(f);



    av_freep(&samples);

    av_frame_free(&frame);

    avcodec_close(c);

    av_free(c);

}
