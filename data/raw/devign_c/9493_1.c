static int rv34_decode_slice(RV34DecContext *r, int end, const uint8_t* buf, int buf_size)
{
    MpegEncContext *s = &r->s;
    GetBitContext *gb = &s->gb;
    int mb_pos;
    int res;
    init_get_bits(&r->s.gb, buf, buf_size*8);
    res = r->parse_slice_header(r, gb, &r->si);
    if(res < 0){
        av_log(s->avctx, AV_LOG_ERROR, "Incorrect or unknown slice header\n");
        return -1;
    if ((s->mb_x == 0 && s->mb_y == 0) || s->current_picture_ptr==NULL) {
        if(s->width != r->si.width || s->height != r->si.height){
            av_log(s->avctx, AV_LOG_DEBUG, "Changing dimensions to %dx%d\n", r->si.width,r->si.height);
            MPV_common_end(s);
            s->width  = r->si.width;
            s->height = r->si.height;
            avcodec_set_dimensions(s->avctx, s->width, s->height);
            if(MPV_common_init(s) < 0)
                return -1;
            r->intra_types_stride = s->mb_width*4 + 4;
            r->intra_types_hist = av_realloc(r->intra_types_hist, r->intra_types_stride * 4 * 2 * sizeof(*r->intra_types_hist));
            r->intra_types = r->intra_types_hist + r->intra_types_stride * 4;
            r->mb_type = av_realloc(r->mb_type, r->s.mb_stride * r->s.mb_height * sizeof(*r->mb_type));
            r->cbp_luma   = av_realloc(r->cbp_luma,   r->s.mb_stride * r->s.mb_height * sizeof(*r->cbp_luma));
            r->cbp_chroma = av_realloc(r->cbp_chroma, r->s.mb_stride * r->s.mb_height * sizeof(*r->cbp_chroma));
            r->deblock_coefs = av_realloc(r->deblock_coefs, r->s.mb_stride * r->s.mb_height * sizeof(*r->deblock_coefs));
        s->pict_type = r->si.type ? r->si.type : AV_PICTURE_TYPE_I;
        if(MPV_frame_start(s, s->avctx) < 0)
            return -1;
        ff_er_frame_start(s);
        if (!r->tmp_b_block_base || s->width != r->si.width || s->height != r->si.height) {
            int i;
            av_free(r->tmp_b_block_base); //realloc() doesn't guarantee alignment
            r->tmp_b_block_base = av_malloc(s->linesize * 48);
            for (i = 0; i < 2; i++)
                r->tmp_b_block_y[i] = r->tmp_b_block_base + i * 16 * s->linesize;
            for (i = 0; i < 4; i++)
                r->tmp_b_block_uv[i] = r->tmp_b_block_base + 32 * s->linesize
                                       + (i >> 1) * 8 * s->uvlinesize + (i & 1) * 16;
        r->cur_pts = r->si.pts;
        if(s->pict_type != AV_PICTURE_TYPE_B){
            r->last_pts = r->next_pts;
            r->next_pts = r->cur_pts;
        }else{
            int refdist = GET_PTS_DIFF(r->next_pts, r->last_pts);
            int dist0   = GET_PTS_DIFF(r->cur_pts,  r->last_pts);
            int dist1   = GET_PTS_DIFF(r->next_pts, r->cur_pts);
            if(!refdist){
                r->weight1 = r->weight2 = 8192;
            }else{
                r->weight1 = (dist0 << 14) / refdist;
                r->weight2 = (dist1 << 14) / refdist;
        s->mb_x = s->mb_y = 0;
    r->si.end = end;
    s->qscale = r->si.quant;
    r->bits = buf_size*8;
    s->mb_num_left = r->si.end - r->si.start;
    r->s.mb_skip_run = 0;
    mb_pos = s->mb_x + s->mb_y * s->mb_width;
    if(r->si.start != mb_pos){
        av_log(s->avctx, AV_LOG_ERROR, "Slice indicates MB offset %d, got %d\n", r->si.start, mb_pos);
        s->mb_x = r->si.start % s->mb_width;
        s->mb_y = r->si.start / s->mb_width;
    memset(r->intra_types_hist, -1, r->intra_types_stride * 4 * 2 * sizeof(*r->intra_types_hist));
    s->first_slice_line = 1;
    s->resync_mb_x = s->mb_x;
    s->resync_mb_y = s->mb_y;
    ff_init_block_index(s);
    while(!check_slice_end(r, s)) {
        ff_update_block_index(s);
        s->dsp.clear_blocks(s->block[0]);
        if(rv34_decode_macroblock(r, r->intra_types + s->mb_x * 4 + 4) < 0){
            ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, AC_ERROR|DC_ERROR|MV_ERROR);
            return -1;
        if (++s->mb_x == s->mb_width) {
            s->mb_x = 0;
            s->mb_y++;
            ff_init_block_index(s);
            memmove(r->intra_types_hist, r->intra_types, r->intra_types_stride * 4 * sizeof(*r->intra_types_hist));
            memset(r->intra_types, -1, r->intra_types_stride * 4 * sizeof(*r->intra_types_hist));
            if(r->loop_filter && s->mb_y >= 2)
                r->loop_filter(r, s->mb_y - 2);
        if(s->mb_x == s->resync_mb_x)
            s->first_slice_line=0;
        s->mb_num_left--;
    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, AC_END|DC_END|MV_END);
    return s->mb_y == s->mb_height;