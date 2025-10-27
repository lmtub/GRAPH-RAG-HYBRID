static int mjpeg_decode_sof0(MJpegDecodeContext *s,

                             UINT8 *buf, int buf_size)

{

    int len, nb_components, i, width, height;



    init_get_bits(&s->gb, buf, buf_size);



    /* XXX: verify len field validity */

    len = get_bits(&s->gb, 16);

    /* only 8 bits/component accepted */

    if (get_bits(&s->gb, 8) != 8)

        return -1;

    height = get_bits(&s->gb, 16);

    width = get_bits(&s->gb, 16);



    nb_components = get_bits(&s->gb, 8);

    if (nb_components <= 0 ||

        nb_components > MAX_COMPONENTS)

        return -1;

    s->nb_components = nb_components;

    s->h_max = 1;

    s->v_max = 1;

    for(i=0;i<nb_components;i++) {

        /* component id */

        s->component_id[i] = get_bits(&s->gb, 8) - 1;

        s->h_count[i] = get_bits(&s->gb, 4);

        s->v_count[i] = get_bits(&s->gb, 4);

        /* compute hmax and vmax (only used in interleaved case) */

        if (s->h_count[i] > s->h_max)

            s->h_max = s->h_count[i];

        if (s->v_count[i] > s->v_max)

            s->v_max = s->v_count[i];

        s->quant_index[i] = get_bits(&s->gb, 8);

        if (s->quant_index[i] >= 4)

            return -1;

        dprintf("component %d %d:%d\n", i, s->h_count[i], s->v_count[i]);

    }



    /* if different size, realloc/alloc picture */

    /* XXX: also check h_count and v_count */

    if (width != s->width || height != s->height) {

        for(i=0;i<MAX_COMPONENTS;i++) {

            free(s->current_picture[i]);

            s->current_picture[i] = NULL;

        }

        s->width = width;

        s->height = height;

        /* test interlaced mode */

        if (s->first_picture &&

            s->org_height != 0 &&

            s->height < ((s->org_height * 3) / 4)) {

            s->interlaced = 1;

            s->bottom_field = 0;

        }



        for(i=0;i<nb_components;i++) {

            int w, h;

            w = (s->width  + 8 * s->h_max - 1) / (8 * s->h_max);

            h = (s->height + 8 * s->v_max - 1) / (8 * s->v_max);

            w = w * 8 * s->h_count[i];

            h = h * 8 * s->v_count[i];

            if (s->interlaced)

                w *= 2;

            s->linesize[i] = w;

            /* memory test is done in mjpeg_decode_sos() */

            s->current_picture[i] = av_mallocz(w * h);

        }

        s->first_picture = 0;

    }



    if (len != 8+(3*nb_components))

	dprintf("decode_sof0: error, len(%d) mismatch\n", len);

    

    return 0;

}
