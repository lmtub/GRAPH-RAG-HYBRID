static int ljpeg_decode_rgb_scan(MJpegDecodeContext *s, int nb_components, int predictor, int point_transform)
{
    int i, mb_x, mb_y;
    uint16_t (*buffer)[4];
    int left[4], top[4], topleft[4];
    const int linesize = s->linesize[0];
    const int mask     = (1 << s->bits) - 1;
    int resync_mb_y = 0;
    int resync_mb_x = 0;
    s->restart_count = s->restart_interval;
    av_fast_malloc(&s->ljpeg_buffer, &s->ljpeg_buffer_size,
                   (unsigned)s->mb_width * 4 * sizeof(s->ljpeg_buffer[0][0]));
    buffer = s->ljpeg_buffer;
    for (i = 0; i < 4; i++)
        buffer[0][i] = 1 << (s->bits - 1);
    for (mb_y = 0; mb_y < s->mb_height; mb_y++) {
        uint8_t *ptr = s->picture.data[0] + (linesize * mb_y);
        if (s->interlaced && s->bottom_field)
            ptr += linesize >> 1;
        for (i = 0; i < 4; i++)
            top[i] = left[i] = topleft[i] = buffer[0][i];
        for (mb_x = 0; mb_x < s->mb_width; mb_x++) {
            int modified_predictor = predictor;
            if (s->restart_interval && !s->restart_count){
                s->restart_count = s->restart_interval;
                resync_mb_x = mb_x;
                resync_mb_y = mb_y;
                for(i=0; i<4; i++)
                    top[i] = left[i]= topleft[i]= 1 << (s->bits - 1);
            }
            if (mb_y == resync_mb_y || mb_y == resync_mb_y+1 && mb_x < resync_mb_x || !mb_x)
                modified_predictor = 1;
            for (i=0;i<nb_components;i++) {
                int pred, dc;
                topleft[i] = top[i];
                top[i]     = buffer[mb_x][i];
                PREDICT(pred, topleft[i], top[i], left[i], modified_predictor);
                dc = mjpeg_decode_dc(s, s->dc_index[i]);
                if(dc == 0xFFFF)
                    return -1;
                left[i] = buffer[mb_x][i] =
                    mask & (pred + (dc << point_transform));
            }
            if (s->restart_interval && !--s->restart_count) {
                align_get_bits(&s->gb);
                skip_bits(&s->gb, 16); /* skip RSTn */
            }
        }
        if (s->nb_components == 4) {
            for(i=0; i<nb_components; i++) {
                int c= s->comp_index[i];
                for(mb_x = 0; mb_x < s->mb_width; mb_x++) {
                    ptr[4*mb_x+3-c] = buffer[mb_x][i];
                }
            }
        } else if (s->rct) {
            for (mb_x = 0; mb_x < s->mb_width; mb_x++) {
                ptr[3*mb_x + 1] = buffer[mb_x][0] - ((buffer[mb_x][1] + buffer[mb_x][2] - 0x200) >> 2);
                ptr[3*mb_x + 0] = buffer[mb_x][1] + ptr[3*mb_x + 1];
                ptr[3*mb_x + 2] = buffer[mb_x][2] + ptr[3*mb_x + 1];
            }
        } else if (s->pegasus_rct) {
            for (mb_x = 0; mb_x < s->mb_width; mb_x++) {
                ptr[3*mb_x + 1] = buffer[mb_x][0] - ((buffer[mb_x][1] + buffer[mb_x][2]) >> 2);
                ptr[3*mb_x + 0] = buffer[mb_x][1] + ptr[3*mb_x + 1];
                ptr[3*mb_x + 2] = buffer[mb_x][2] + ptr[3*mb_x + 1];
            }
        } else {
            for(i=0; i<nb_components; i++) {
                int c= s->comp_index[i];
                for(mb_x = 0; mb_x < s->mb_width; mb_x++) {
                    ptr[3*mb_x+2-c] = buffer[mb_x][i];
                }
            }
        }
    }
    return 0;
}