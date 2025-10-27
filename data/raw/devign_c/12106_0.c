static void rv34_pred_mv_rv3(RV34DecContext *r, int block_type, int dir)

{

    MpegEncContext *s = &r->s;

    int mv_pos = s->mb_x * 2 + s->mb_y * 2 * s->b8_stride;

    int A[2] = {0}, B[2], C[2];

    int i, j, k;

    int mx, my;

    int avail_index = avail_indexes[0];



    if(r->avail_cache[avail_index - 1]){

        A[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - 1][0];

        A[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - 1][1];

    }

    if(r->avail_cache[avail_index - 4]){

        B[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride][0];

        B[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride][1];

    }else{

        B[0] = A[0];

        B[1] = A[1];

    }

    if(!r->avail_cache[avail_index - 4 + 2]){

        if(r->avail_cache[avail_index - 4] && (r->avail_cache[avail_index - 1])){

            C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride - 1][0];

            C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride - 1][1];

        }else{

            C[0] = A[0];

            C[1] = A[1];

        }

    }else{

        C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride + 2][0];

        C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride + 2][1];

    }

    mx = mid_pred(A[0], B[0], C[0]);

    my = mid_pred(A[1], B[1], C[1]);

    mx += r->dmv[0][0];

    my += r->dmv[0][1];

    for(j = 0; j < 2; j++){

        for(i = 0; i < 2; i++){

            for(k = 0; k < 2; k++){

                s->current_picture_ptr->f.motion_val[k][mv_pos + i + j*s->b8_stride][0] = mx;

                s->current_picture_ptr->f.motion_val[k][mv_pos + i + j*s->b8_stride][1] = my;

            }

        }

    }

}
