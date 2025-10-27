static int mp_decode_frame(MPADecodeContext *s, 

                           short *samples)

{

    int i, nb_frames, ch;

    short *samples_ptr;



    init_get_bits(&s->gb, s->inbuf + HEADER_SIZE, 

                  s->inbuf_ptr - s->inbuf - HEADER_SIZE);

    

    /* skip error protection field */

    if (s->error_protection)

        get_bits(&s->gb, 16);



    dprintf("frame %d:\n", s->frame_count);

    switch(s->layer) {

    case 1:

        nb_frames = mp_decode_layer1(s);

        break;

    case 2:

        nb_frames = mp_decode_layer2(s);

        break;

    case 3:

    default:

        nb_frames = mp_decode_layer3(s);

        break;

    }

#if defined(DEBUG)

    for(i=0;i<nb_frames;i++) {

        for(ch=0;ch<s->nb_channels;ch++) {

            int j;

            printf("%d-%d:", i, ch);

            for(j=0;j<SBLIMIT;j++)

                printf(" %0.6f", (double)s->sb_samples[ch][i][j] / FRAC_ONE);

            printf("\n");

        }

    }

#endif

    /* apply the synthesis filter */

    for(ch=0;ch<s->nb_channels;ch++) {

        samples_ptr = samples + ch;

        for(i=0;i<nb_frames;i++) {

            synth_filter(s, ch, samples_ptr, s->nb_channels,

                         s->sb_samples[ch][i]);

            samples_ptr += 32 * s->nb_channels;

        }

    }

#ifdef DEBUG

    s->frame_count++;        

#endif

    return nb_frames * 32 * sizeof(short) * s->nb_channels;

}
