static int quantize(CinepakEncContext *s, int h, AVPicture *pict, int v1mode, int size, int v4, strip_info *info)

{

    int x, y, i, j, k, x2, y2, x3, y3, plane, shift;

    int entry_size = s->pix_fmt == AV_PIX_FMT_YUV420P ? 6 : 4;

    int *codebook = v1mode ? info->v1_codebook : info->v4_codebook;

    int64_t total_error = 0;

    uint8_t vq_pict_buf[(MB_AREA*3)/2];

    AVPicture sub_pict, vq_pict;



    for(i = y = 0; y < h; y += MB_SIZE) {

        for(x = 0; x < s->w; x += MB_SIZE, i += v1mode ? 1 : 4) {

            int *base = s->codebook_input + i*entry_size;



            if(v1mode) {

                //subsample

                for(j = y2 = 0; y2 < entry_size; y2 += 2) {

                    for(x2 = 0; x2 < 4; x2 += 2, j++) {

                        plane = y2 < 4 ? 0 : 1 + (x2 >> 1);

                        shift = y2 < 4 ? 0 : 1;

                        x3 = shift ? 0 : x2;

                        y3 = shift ? 0 : y2;

                        base[j] = (pict->data[plane][((x+x3) >> shift) +      ((y+y3) >> shift)      * pict->linesize[plane]] +

                                   pict->data[plane][((x+x3) >> shift) + 1 +  ((y+y3) >> shift)      * pict->linesize[plane]] +

                                   pict->data[plane][((x+x3) >> shift) +     (((y+y3) >> shift) + 1) * pict->linesize[plane]] +

                                   pict->data[plane][((x+x3) >> shift) + 1 + (((y+y3) >> shift) + 1) * pict->linesize[plane]]) >> 2;

                    }

                }

            } else {

                //copy

                for(j = y2 = 0; y2 < MB_SIZE; y2 += 2) {

                    for(x2 = 0; x2 < MB_SIZE; x2 += 2) {

                        for(k = 0; k < entry_size; k++, j++) {

                            plane = k >= 4 ? k - 3 : 0;



                            if(k >= 4) {

                                x3 = (x+x2) >> 1;

                                y3 = (y+y2) >> 1;

                            } else {

                                x3 = x + x2 + (k & 1);

                                y3 = y + y2 + (k >> 1);

                            }



                            base[j] = pict->data[plane][x3 + y3*pict->linesize[plane]];

                        }

                    }

                }

            }

        }

    }



    ff_init_elbg(s->codebook_input, entry_size, i, codebook, size, 1, s->codebook_closest, &s->randctx);

    ff_do_elbg(s->codebook_input, entry_size, i, codebook, size, 1, s->codebook_closest, &s->randctx);



    //setup vq_pict, which contains a single MB

    vq_pict.data[0] = vq_pict_buf;

    vq_pict.linesize[0] = MB_SIZE;

    vq_pict.data[1] = &vq_pict_buf[MB_AREA];

    vq_pict.data[2] = vq_pict.data[1] + (MB_AREA >> 2);

    vq_pict.linesize[1] = vq_pict.linesize[2] = MB_SIZE >> 1;



    //copy indices

    for(i = j = y = 0; y < h; y += MB_SIZE) {

        for(x = 0; x < s->w; x += MB_SIZE, j++, i += v1mode ? 1 : 4) {

            mb_info *mb = &s->mb[j];



            //point sub_pict to current MB

            get_sub_picture(s, x, y, pict, &sub_pict);



            if(v1mode) {

                mb->v1_vector = s->codebook_closest[i];



                //fill in vq_pict with V1 data

                decode_v1_vector(s, &vq_pict, mb, info);



                mb->v1_error = compute_mb_distortion(s, &sub_pict, &vq_pict);

                total_error += mb->v1_error;

            } else {

                for(k = 0; k < 4; k++)

                    mb->v4_vector[v4][k] = s->codebook_closest[i+k];



                //fill in vq_pict with V4 data

                decode_v4_vector(s, &vq_pict, mb->v4_vector[v4], info);



                mb->v4_error[v4] = compute_mb_distortion(s, &sub_pict, &vq_pict);

                total_error += mb->v4_error[v4];

            }

        }

    }



    //av_log(s->avctx, AV_LOG_INFO, "mode %i size %i i %i error %li\n", v1mode, size, i, total_error);



    return 0;

}
