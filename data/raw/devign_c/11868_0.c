static void mpeg_decode_extension(AVCodecContext *avctx, 

                                  UINT8 *buf, int buf_size)

{

    Mpeg1Context *s1 = avctx->priv_data;

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    int ext_type;



    init_get_bits(&s->gb, buf, buf_size);

    

    ext_type = get_bits(&s->gb, 4);

    switch(ext_type) {

    case 0x1:

        /* sequence ext */

        mpeg_decode_sequence_extension(s);

        break;

    case 0x3:

        /* quant matrix extension */

        mpeg_decode_quant_matrix_extension(s);

        break;

    case 0x8:

        /* picture extension */

        mpeg_decode_picture_coding_extension(s);

        break;

    }

}
