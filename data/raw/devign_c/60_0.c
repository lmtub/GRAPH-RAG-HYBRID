static int mpeg1_decode_sequence(AVCodecContext *avctx, 

                                 UINT8 *buf, int buf_size)

{

    Mpeg1Context *s1 = avctx->priv_data;

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    int width, height, i, v, j;

    float aspect;



    init_get_bits(&s->gb, buf, buf_size);



    width = get_bits(&s->gb, 12);

    height = get_bits(&s->gb, 12);

    s->aspect_ratio_info= get_bits(&s->gb, 4);

    if(!s->mpeg2){

        aspect= mpeg1_aspect[s->aspect_ratio_info];

        if(aspect!=0.0) avctx->aspect_ratio= width/(aspect*height);

    }



    s->frame_rate_index = get_bits(&s->gb, 4);

    if (s->frame_rate_index == 0)

        return -1;

    s->bit_rate = get_bits(&s->gb, 18) * 400;

    if (get_bits1(&s->gb) == 0) /* marker */

        return -1;

    if (width <= 0 || height <= 0 ||

        (width % 2) != 0 || (height % 2) != 0)

        return -1;

    if (width != s->width ||

        height != s->height) {

        /* start new mpeg1 context decoding */

        s->out_format = FMT_MPEG1;

        if (s1->mpeg_enc_ctx_allocated) {

            MPV_common_end(s);

        }

        s->width = width;

        s->height = height;

        avctx->has_b_frames= 1;

        s->avctx = avctx;

        avctx->width = width;

        avctx->height = height;

        if (s->frame_rate_index >= 9) {

            /* at least give a valid frame rate (some old mpeg1 have this) */

            avctx->frame_rate = 25 * FRAME_RATE_BASE;

        } else {

            avctx->frame_rate = frame_rate_tab[s->frame_rate_index];

        }

        s->frame_rate = avctx->frame_rate;

        avctx->bit_rate = s->bit_rate;

        

        if (MPV_common_init(s) < 0)

            return -1;

        s1->mpeg_enc_ctx_allocated = 1;

    }



    skip_bits(&s->gb, 10); /* vbv_buffer_size */

    skip_bits(&s->gb, 1);



    /* get matrix */

    if (get_bits1(&s->gb)) {

        for(i=0;i<64;i++) {

            v = get_bits(&s->gb, 8);

            j = s->intra_scantable.permutated[i];

            s->intra_matrix[j] = v;

            s->chroma_intra_matrix[j] = v;

        }

#ifdef DEBUG

        dprintf("intra matrix present\n");

        for(i=0;i<64;i++)

            dprintf(" %d", s->intra_matrix[s->intra_scantable.permutated[i]]);

        printf("\n");

#endif

    } else {

        for(i=0;i<64;i++) {

            int j= s->idct_permutation[i];

            v = ff_mpeg1_default_intra_matrix[i];

            s->intra_matrix[j] = v;

            s->chroma_intra_matrix[j] = v;

        }

    }

    if (get_bits1(&s->gb)) {

        for(i=0;i<64;i++) {

            v = get_bits(&s->gb, 8);

            j = s->intra_scantable.permutated[i];

            s->inter_matrix[j] = v;

            s->chroma_inter_matrix[j] = v;

        }

#ifdef DEBUG

        dprintf("non intra matrix present\n");

        for(i=0;i<64;i++)

            dprintf(" %d", s->inter_matrix[s->intra_scantable.permutated[i]]);

        printf("\n");

#endif

    } else {

        for(i=0;i<64;i++) {

            int j= s->idct_permutation[i];

            v = ff_mpeg1_default_non_intra_matrix[i];

            s->inter_matrix[j] = v;

            s->chroma_inter_matrix[j] = v;

        }

    }



    /* we set mpeg2 parameters so that it emulates mpeg1 */

    s->progressive_sequence = 1;

    s->progressive_frame = 1;

    s->picture_structure = PICT_FRAME;

    s->frame_pred_frame_dct = 1;

    s->mpeg2 = 0;

    avctx->sub_id = 1; /* indicates mpeg1 */

    return 0;

}
