static int decode_chunks(AVCodecContext *avctx,
                         AVFrame *picture, int *got_output,
                         const uint8_t *buf, int buf_size)
{
    Mpeg1Context *s = avctx->priv_data;
    MpegEncContext *s2 = &s->mpeg_enc_ctx;
    const uint8_t *buf_ptr = buf;
    const uint8_t *buf_end = buf + buf_size;
    int ret, input_size;
    int last_code = 0, skip_frame = 0;
    int picture_start_code_seen = 0;
    for (;;) {
        /* find next start code */
        uint32_t start_code = -1;
        buf_ptr = avpriv_find_start_code(buf_ptr, buf_end, &start_code);
        if (start_code > 0x1ff) {
            if (!skip_frame) {
                if (HAVE_THREADS && (avctx->active_thread_type & FF_THREAD_SLICE) &&
                    !avctx->hwaccel) {
                    int i;
                    av_assert0(avctx->thread_count > 1);
                    avctx->execute(avctx, slice_decode_thread,  &s2->thread_context[0], NULL, s->slice_count, sizeof(void*));
                    for (i = 0; i < s->slice_count; i++)
                        s2->er.error_count += s2->thread_context[i]->er.error_count;
                ret = slice_end(avctx, picture);
                if (ret < 0)
                    return ret;
                else if (ret) {
                    if (s2->last_picture_ptr || s2->low_delay) //FIXME merge with the stuff in mpeg_decode_slice
                        *got_output = 1;
            s2->pict_type = 0;
            return FFMAX(0, buf_ptr - buf - s2->parse_context.last_index);
        input_size = buf_end - buf_ptr;
        if (avctx->debug & FF_DEBUG_STARTCODE) {
            av_log(avctx, AV_LOG_DEBUG, "%3X at %td left %d\n", start_code, buf_ptr-buf, input_size);
        /* prepare data for next start code */
        switch (start_code) {
        case SEQ_START_CODE:
            if (last_code == 0) {
                mpeg1_decode_sequence(avctx, buf_ptr, input_size);
                if(buf != avctx->extradata)
                    s->sync=1;
            } else {
                av_log(avctx, AV_LOG_ERROR, "ignoring SEQ_START_CODE after %X\n", last_code);
                if (avctx->err_recognition & AV_EF_EXPLODE)
            break;
        case PICTURE_START_CODE:
            if (picture_start_code_seen && s2->picture_structure == PICT_FRAME) {
               /* If it's a frame picture, there can't be more than one picture header.
                  Yet, it does happen and we need to handle it. */
               av_log(avctx, AV_LOG_WARNING, "ignoring extra picture following a frame-picture\n");
               break;
            picture_start_code_seen = 1;
            if (s2->width <= 0 || s2->height <= 0) {
                av_log(avctx, AV_LOG_ERROR, "Invalid frame dimensions %dx%d.\n",
                       s2->width, s2->height);
            if(s->tmpgexs){
                s2->intra_dc_precision= 3;
                s2->intra_matrix[0]= 1;
            if (HAVE_THREADS && (avctx->active_thread_type & FF_THREAD_SLICE) &&
                !avctx->hwaccel && s->slice_count) {
                int i;
                avctx->execute(avctx, slice_decode_thread,
                               s2->thread_context, NULL,
                               s->slice_count, sizeof(void*));
                for (i = 0; i < s->slice_count; i++)
                    s2->er.error_count += s2->thread_context[i]->er.error_count;
                s->slice_count = 0;
            if (last_code == 0 || last_code == SLICE_MIN_START_CODE) {
                ret = mpeg_decode_postinit(avctx);
                if (ret < 0) {
                    av_log(avctx, AV_LOG_ERROR, "mpeg_decode_postinit() failure\n");
                    return ret;
                /* we have a complete image: we try to decompress it */
                if (mpeg1_decode_picture(avctx, buf_ptr, input_size) < 0)
                    s2->pict_type = 0;
                s2->first_slice = 1;
                last_code = PICTURE_START_CODE;
            } else {
                av_log(avctx, AV_LOG_ERROR, "ignoring pic after %X\n", last_code);
                if (avctx->err_recognition & AV_EF_EXPLODE)
            break;
        case EXT_START_CODE:
            init_get_bits(&s2->gb, buf_ptr, input_size*8);
            switch (get_bits(&s2->gb, 4)) {
            case 0x1:
                if (last_code == 0) {
                mpeg_decode_sequence_extension(s);
                } else {
                    av_log(avctx, AV_LOG_ERROR, "ignoring seq ext after %X\n", last_code);
                    if (avctx->err_recognition & AV_EF_EXPLODE)
                break;
            case 0x2:
                mpeg_decode_sequence_display_extension(s);
                break;
            case 0x3:
                mpeg_decode_quant_matrix_extension(s2);
                break;
            case 0x7:
                mpeg_decode_picture_display_extension(s);
                break;
            case 0x8:
                if (last_code == PICTURE_START_CODE) {
                    mpeg_decode_picture_coding_extension(s);
                } else {
                    av_log(avctx, AV_LOG_ERROR, "ignoring pic cod ext after %X\n", last_code);
                    if (avctx->err_recognition & AV_EF_EXPLODE)
                break;
            break;
        case USER_START_CODE:
            mpeg_decode_user_data(avctx, buf_ptr, input_size);
            break;
        case GOP_START_CODE:
            if (last_code == 0) {
                s2->first_field=0;
                mpeg_decode_gop(avctx, buf_ptr, input_size);
                s->sync=1;
            } else {
                av_log(avctx, AV_LOG_ERROR, "ignoring GOP_START_CODE after %X\n", last_code);
                if (avctx->err_recognition & AV_EF_EXPLODE)
            break;
        default:
            if (start_code >= SLICE_MIN_START_CODE &&
                start_code <= SLICE_MAX_START_CODE && last_code == PICTURE_START_CODE) {
                if (s2->progressive_sequence && !s2->progressive_frame) {
                    s2->progressive_frame = 1;
                    av_log(s2->avctx, AV_LOG_ERROR, "interlaced frame in progressive sequence, ignoring\n");
                if (s2->picture_structure == 0 || (s2->progressive_frame && s2->picture_structure != PICT_FRAME)) {
                    av_log(s2->avctx, AV_LOG_ERROR, "picture_structure %d invalid, ignoring\n", s2->picture_structure);
                    s2->picture_structure = PICT_FRAME;
                if (s2->progressive_sequence && !s2->frame_pred_frame_dct) {
                    av_log(s2->avctx, AV_LOG_WARNING, "invalid frame_pred_frame_dct\n");
                if (s2->picture_structure == PICT_FRAME) {
                    s2->first_field = 0;
                    s2->v_edge_pos  = 16 * s2->mb_height;
                } else {
                    s2->first_field ^= 1;
                    s2->v_edge_pos   = 8 * s2->mb_height;
                    memset(s2->mbskip_table, 0, s2->mb_stride * s2->mb_height);
            if (start_code >= SLICE_MIN_START_CODE &&
                start_code <= SLICE_MAX_START_CODE && last_code != 0) {
                const int field_pic = s2->picture_structure != PICT_FRAME;
                int mb_y = start_code - SLICE_MIN_START_CODE;
                last_code = SLICE_MIN_START_CODE;
                if(s2->codec_id != AV_CODEC_ID_MPEG1VIDEO && s2->mb_height > 2800/16)
                    mb_y += (*buf_ptr&0xE0)<<2;
                mb_y <<= field_pic;
                if (s2->picture_structure == PICT_BOTTOM_FIELD)
                    mb_y++;
                if (mb_y >= s2->mb_height) {
                    av_log(s2->avctx, AV_LOG_ERROR, "slice below image (%d >= %d)\n", mb_y, s2->mb_height);
                    return -1;
                if (s2->last_picture_ptr == NULL) {
                /* Skip B-frames if we do not have reference frames and gop is not closed */
                    if (s2->pict_type == AV_PICTURE_TYPE_B) {
                        if (!s2->closed_gop) {
                            skip_frame = 1;
                            break;
                if (s2->pict_type == AV_PICTURE_TYPE_I || (s2->flags2 & CODEC_FLAG2_SHOW_ALL))
                    s->sync=1;
                if (s2->next_picture_ptr == NULL) {
                /* Skip P-frames if we do not have a reference frame or we have an invalid header. */
                    if (s2->pict_type == AV_PICTURE_TYPE_P && !s->sync) {
                        skip_frame = 1;
                        break;
                if ((avctx->skip_frame >= AVDISCARD_NONREF && s2->pict_type == AV_PICTURE_TYPE_B) ||
                    (avctx->skip_frame >= AVDISCARD_NONKEY && s2->pict_type != AV_PICTURE_TYPE_I) ||
                     avctx->skip_frame >= AVDISCARD_ALL) {
                    skip_frame = 1;
                    break;
                if (!s->mpeg_enc_ctx_allocated)
                    break;
                if (s2->codec_id == AV_CODEC_ID_MPEG2VIDEO) {
                    if (mb_y < avctx->skip_top || mb_y >= s2->mb_height - avctx->skip_bottom)
                        break;
                if (!s2->pict_type) {
                    av_log(avctx, AV_LOG_ERROR, "Missing picture start code\n");
                    if (avctx->err_recognition & AV_EF_EXPLODE)
                    break;
                if (s2->first_slice) {
                    skip_frame = 0;
                    s2->first_slice = 0;
                    if (mpeg_field_start(s2, buf, buf_size) < 0)
                        return -1;
                if (!s2->current_picture_ptr) {
                    av_log(avctx, AV_LOG_ERROR, "current_picture not initialized\n");
                if (HAVE_THREADS && (avctx->active_thread_type & FF_THREAD_SLICE) &&
                    !avctx->hwaccel) {
                    int threshold = (s2->mb_height * s->slice_count +
                                     s2->slice_context_count / 2) /
                                    s2->slice_context_count;
                    av_assert0(avctx->thread_count > 1);
                    if (threshold <= mb_y) {
                        MpegEncContext *thread_context = s2->thread_context[s->slice_count];
                        thread_context->start_mb_y = mb_y;
                        thread_context->end_mb_y   = s2->mb_height;
                        if (s->slice_count) {
                            s2->thread_context[s->slice_count-1]->end_mb_y = mb_y;
                            ret = ff_update_duplicate_context(thread_context,
                                                              s2);
                            if (ret < 0)
                                return ret;
                        init_get_bits(&thread_context->gb, buf_ptr, input_size*8);
                        s->slice_count++;
                    buf_ptr += 2; // FIXME add minimum number of bytes per slice
                } else {
                    ret = mpeg_decode_slice(s2, mb_y, &buf_ptr, input_size);
                    emms_c();
                    if (ret < 0) {
                        if (avctx->err_recognition & AV_EF_EXPLODE)
                            return ret;
                        if (s2->resync_mb_x >= 0 && s2->resync_mb_y >= 0)
                            ff_er_add_slice(&s2->er, s2->resync_mb_x, s2->resync_mb_y, s2->mb_x, s2->mb_y, ER_AC_ERROR | ER_DC_ERROR | ER_MV_ERROR);
                    } else {
                        ff_er_add_slice(&s2->er, s2->resync_mb_x, s2->resync_mb_y, s2->mb_x-1, s2->mb_y, ER_AC_END | ER_DC_END | ER_MV_END);
            break;