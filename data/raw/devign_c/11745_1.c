void audio_decode_example(const char *outfilename, const char *filename)

{

    AVCodec *codec;

    AVCodecContext *c= NULL;

    int out_size, size, len;

    FILE *f, *outfile;

    uint8_t *outbuf;

    uint8_t inbuf[INBUF_SIZE + FF_INPUT_BUFFER_PADDING_SIZE], *inbuf_ptr;



    printf("Audio decoding\n");

    

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged mpeg streams) */

    memset(inbuf + INBUF_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);



    /* find the mpeg audio decoder */

    codec = avcodec_find_decoder(CODEC_ID_MP2);

    if (!codec) {

        fprintf(stderr, "codec not found\n");

        exit(1);

    }



    c= avcodec_alloc_context();



    /* open it */

    if (avcodec_open(c, codec) < 0) {

        fprintf(stderr, "could not open codec\n");

        exit(1);

    }

    

    outbuf = malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE);



    f = fopen(filename, "r");

    if (!f) {

        fprintf(stderr, "could not open %s\n", filename);

        exit(1);

    }

    outfile = fopen(outfilename, "w");

    if (!outfile) {

        free(c);

        exit(1);

    }

        

    /* decode until eof */

    inbuf_ptr = inbuf;

    for(;;) {

        size = fread(inbuf, 1, INBUF_SIZE, f);

        if (size == 0)

            break;



        inbuf_ptr = inbuf;

        while (size > 0) {

            len = avcodec_decode_audio(c, (short *)outbuf, &out_size, 

                                       inbuf_ptr, size);

            if (len < 0) {

                fprintf(stderr, "Error while decoding\n");

                exit(1);

            }

            if (out_size > 0) {

                /* if a frame has been decoded, output it */

                fwrite(outbuf, 1, out_size, outfile);

            }

            size -= len;

            inbuf_ptr += len;

        }

    }



    fclose(outfile);

    fclose(f);

    free(outbuf);



    avcodec_close(c);

    free(c);

}
