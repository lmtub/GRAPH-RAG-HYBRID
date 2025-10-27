static void hl_decode_mb(H264Context *h){

    MpegEncContext * const s = &h->s;

    const int mb_xy= h->mb_xy;

    const int mb_type= s->current_picture.mb_type[mb_xy];

    int is_complex = h->is_complex || IS_INTRA_PCM(mb_type) || s->qscale == 0;



    if(ENABLE_H264_ENCODER && !s->decode)

        return;



    if (is_complex)

        hl_decode_mb_complex(h);

    else hl_decode_mb_simple(h);

}
