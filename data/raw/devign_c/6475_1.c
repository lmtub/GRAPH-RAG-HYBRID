static int vp3_decode_frame(AVCodecContext *avctx, 

                            void *data, int *data_size,

                            uint8_t *buf, int buf_size)

{

    Vp3DecodeContext *s = avctx->priv_data;

    GetBitContext gb;

    static int counter = 0;



    *data_size = 0;



    init_get_bits(&gb, buf, buf_size * 8);



    s->keyframe = get_bits(&gb, 1);

    s->keyframe ^= 1;

    skip_bits(&gb, 1);

    s->last_quality_index = s->quality_index;

    s->quality_index = get_bits(&gb, 6);

    if (s->quality_index != s->last_quality_index)

        init_dequantizer(s);



    debug_vp3(" VP3 frame #%d: Q index = %d", counter, s->quality_index);

    counter++;



    if (s->keyframe) {

        if ((s->golden_frame.data[0]) &&

            (s->last_frame.data[0] == s->golden_frame.data[0]))

            avctx->release_buffer(avctx, &s->golden_frame);

        else if (s->last_frame.data[0])

            avctx->release_buffer(avctx, &s->last_frame);



        s->golden_frame.reference = 0;

        if(avctx->get_buffer(avctx, &s->golden_frame) < 0) {

            printf("vp3: get_buffer() failed\n");

            return -1;

        }



        /* golden frame is also the current frame */

        memcpy(&s->current_frame, &s->golden_frame, sizeof(AVFrame));



        /* time to figure out pixel addresses? */

        if (!s->pixel_addresses_inited)

            vp3_calculate_pixel_addresses(s);



    } else {



        /* allocate a new current frame */

        s->current_frame.reference = 0;

        if(avctx->get_buffer(avctx, &s->current_frame) < 0) {

            printf("vp3: get_buffer() failed\n");

            return -1;

        }



    }



    if (s->keyframe) {

      debug_vp3(", keyframe\n");

      /* skip the other 2 header bytes for now */

      skip_bits(&gb, 16);

    } else

      debug_vp3("\n");



    init_frame(s, &gb);



#if KEYFRAMES_ONLY

if (!s->keyframe) {



    memcpy(s->current_frame.data[0], s->golden_frame.data[0],

        s->current_frame.linesize[0] * s->height);

    memcpy(s->current_frame.data[1], s->golden_frame.data[1],

        s->current_frame.linesize[1] * s->height / 2);

    memcpy(s->current_frame.data[2], s->golden_frame.data[2],

        s->current_frame.linesize[2] * s->height / 2);



} else {

#endif



    if (unpack_superblocks(s, &gb) ||

        unpack_modes(s, &gb) ||

        unpack_vectors(s, &gb) ||

        unpack_dct_coeffs(s, &gb)) {



        printf("  vp3: could not decode frame\n");

        return -1;

    }



    reverse_dc_prediction(s, 0, s->fragment_width, s->fragment_height);

    reverse_dc_prediction(s, s->u_fragment_start,

        s->fragment_width / 2, s->fragment_height / 2);

    reverse_dc_prediction(s, s->v_fragment_start,

        s->fragment_width / 2, s->fragment_height / 2);



    render_fragments(s, 0, s->width, s->height, 0);

    render_fragments(s, s->u_fragment_start, s->width / 2, s->height / 2, 1);

    render_fragments(s, s->v_fragment_start, s->width / 2, s->height / 2, 2);



#if KEYFRAMES_ONLY

}

#endif



    *data_size=sizeof(AVFrame);

    *(AVFrame*)data= s->current_frame;



    /* release the last frame, if it is allocated and if it is not the

     * golden frame */

    if ((s->last_frame.data[0]) &&

        (s->last_frame.data[0] != s->golden_frame.data[0]))

        avctx->release_buffer(avctx, &s->last_frame);



    /* shuffle frames (last = current) */

    memcpy(&s->last_frame, &s->current_frame, sizeof(AVFrame));



    return buf_size;

}
