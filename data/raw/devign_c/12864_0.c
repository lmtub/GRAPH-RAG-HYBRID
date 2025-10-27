void ff_cavs_init_top_lines(AVSContext *h) {

    /* alloc top line of predictors */

    h->top_qp       = av_malloc( h->mb_width);

    h->top_mv[0]    = av_malloc((h->mb_width*2+1)*sizeof(cavs_vector));

    h->top_mv[1]    = av_malloc((h->mb_width*2+1)*sizeof(cavs_vector));

    h->top_pred_Y   = av_malloc( h->mb_width*2*sizeof(*h->top_pred_Y));

    h->top_border_y = av_malloc((h->mb_width+1)*16);

    h->top_border_u = av_malloc( h->mb_width * 10);

    h->top_border_v = av_malloc( h->mb_width * 10);



    /* alloc space for co-located MVs and types */

    h->col_mv       = av_malloc( h->mb_width*h->mb_height*4*sizeof(cavs_vector));

    h->col_type_base = av_malloc(h->mb_width*h->mb_height);

    h->block        = av_mallocz(64*sizeof(DCTELEM));

}
