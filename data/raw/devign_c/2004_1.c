static void mpeg_decode_quant_matrix_extension(MpegEncContext *s)

{

    int i, v, j;



    dprintf("matrix extension\n");



    if (get_bits1(&s->gb)) {

        for(i=0;i<64;i++) {

            v = get_bits(&s->gb, 8);

            j = zigzag_direct[i];

            s->intra_matrix[j] = v;

            s->chroma_intra_matrix[j] = v;

        }

    }

    if (get_bits1(&s->gb)) {

        for(i=0;i<64;i++) {

            v = get_bits(&s->gb, 8);

            j = zigzag_direct[i];

            s->non_intra_matrix[j] = v;

            s->chroma_non_intra_matrix[j] = v;

        }

    }

    if (get_bits1(&s->gb)) {

        for(i=0;i<64;i++) {

            v = get_bits(&s->gb, 8);

            j = zigzag_direct[i];

            s->chroma_intra_matrix[j] = v;

        }

    }

    if (get_bits1(&s->gb)) {

        for(i=0;i<64;i++) {

            v = get_bits(&s->gb, 8);

            j = zigzag_direct[i];

            s->chroma_non_intra_matrix[j] = v;

        }

    }

}
