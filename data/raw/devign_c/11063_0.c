static av_cold int qtrle_encode_end(AVCodecContext *avctx)

{

    QtrleEncContext *s = avctx->priv_data;



    av_frame_free(&avctx->coded_frame);



    avpicture_free(&s->previous_frame);

    av_free(s->rlecode_table);

    av_free(s->length_table);

    av_free(s->skip_table);

    return 0;

}
