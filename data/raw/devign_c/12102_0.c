static int dv_decode_video_segment(AVCodecContext *avctx, void *arg)

{

    DVVideoContext *s = avctx->priv_data;

    DVwork_chunk *work_chunk = arg;

    int quant, dc, dct_mode, class1, j;

    int mb_index, mb_x, mb_y, last_index;

    int y_stride, linesize;

    DCTELEM *block, *block1;

    int c_offset;

    uint8_t *y_ptr;

    const uint8_t *buf_ptr;

    PutBitContext pb, vs_pb;

    GetBitContext gb;

    BlockInfo mb_data[5 * DV_MAX_BPM], *mb, *mb1;

    LOCAL_ALIGNED_16(DCTELEM, sblock, [5*DV_MAX_BPM], [64]);

    LOCAL_ALIGNED_16(uint8_t, mb_bit_buffer, [  80 + FF_INPUT_BUFFER_PADDING_SIZE]); /* allow some slack */

    LOCAL_ALIGNED_16(uint8_t, vs_bit_buffer, [5*80 + FF_INPUT_BUFFER_PADDING_SIZE]); /* allow some slack */

    const int log2_blocksize = 3;

    int is_field_mode[5];



    assert((((int)mb_bit_buffer) & 7) == 0);

    assert((((int)vs_bit_buffer) & 7) == 0);



    memset(sblock, 0, 5*DV_MAX_BPM*sizeof(*sblock));



    /* pass 1: read DC and AC coefficients in blocks */

    buf_ptr = &s->buf[work_chunk->buf_offset*80];

    block1  = &sblock[0][0];

    mb1     = mb_data;

    init_put_bits(&vs_pb, vs_bit_buffer, 5 * 80);

    for (mb_index = 0; mb_index < 5; mb_index++, mb1 += s->sys->bpm, block1 += s->sys->bpm * 64) {

        /* skip header */

        quant = buf_ptr[3] & 0x0f;

        buf_ptr += 4;

        init_put_bits(&pb, mb_bit_buffer, 80);

        mb    = mb1;

        block = block1;

        is_field_mode[mb_index] = 0;

        for (j = 0; j < s->sys->bpm; j++) {

            last_index = s->sys->block_sizes[j];

            init_get_bits(&gb, buf_ptr, last_index);



            /* get the DC */

            dc       = get_sbits(&gb, 9);

            dct_mode = get_bits1(&gb);

            class1   = get_bits(&gb, 2);

            if (DV_PROFILE_IS_HD(s->sys)) {

                mb->idct_put     = s->idct_put[0];

                mb->scan_table   = s->dv_zigzag[0];

                mb->factor_table = &s->sys->idct_factor[(j >= 4)*4*16*64 + class1*16*64 + quant*64];

                is_field_mode[mb_index] |= !j && dct_mode;

            } else {

                mb->idct_put     = s->idct_put[dct_mode && log2_blocksize == 3];

                mb->scan_table   = s->dv_zigzag[dct_mode];

                mb->factor_table = &s->sys->idct_factor[(class1 == 3)*2*22*64 + dct_mode*22*64 +

                                                        (quant + ff_dv_quant_offset[class1])*64];

            }

            dc = dc << 2;

            /* convert to unsigned because 128 is not added in the

               standard IDCT */

            dc += 1024;

            block[0] = dc;

            buf_ptr += last_index >> 3;

            mb->pos               = 0;

            mb->partial_bit_count = 0;



            av_dlog(avctx, "MB block: %d, %d ", mb_index, j);

            dv_decode_ac(&gb, mb, block);



            /* write the remaining bits in a new buffer only if the

               block is finished */

            if (mb->pos >= 64)

                bit_copy(&pb, &gb);



            block += 64;

            mb++;

        }



        /* pass 2: we can do it just after */

        av_dlog(avctx, "***pass 2 size=%d MB#=%d\n", put_bits_count(&pb), mb_index);

        block = block1;

        mb    = mb1;

        init_get_bits(&gb, mb_bit_buffer, put_bits_count(&pb));

        put_bits32(&pb, 0); // padding must be zeroed

        flush_put_bits(&pb);

        for (j = 0; j < s->sys->bpm; j++, block += 64, mb++) {

            if (mb->pos < 64 && get_bits_left(&gb) > 0) {

                dv_decode_ac(&gb, mb, block);

                /* if still not finished, no need to parse other blocks */

                if (mb->pos < 64)

                    break;

            }

        }

        /* all blocks are finished, so the extra bytes can be used at

           the video segment level */

        if (j >= s->sys->bpm)

            bit_copy(&vs_pb, &gb);

    }



    /* we need a pass over the whole video segment */

    av_dlog(avctx, "***pass 3 size=%d\n", put_bits_count(&vs_pb));

    block = &sblock[0][0];

    mb    = mb_data;

    init_get_bits(&gb, vs_bit_buffer, put_bits_count(&vs_pb));

    put_bits32(&vs_pb, 0); // padding must be zeroed

    flush_put_bits(&vs_pb);

    for (mb_index = 0; mb_index < 5; mb_index++) {

        for (j = 0; j < s->sys->bpm; j++) {

            if (mb->pos < 64) {

                av_dlog(avctx, "start %d:%d\n", mb_index, j);

                dv_decode_ac(&gb, mb, block);

            }

            if (mb->pos >= 64 && mb->pos < 127)

                av_log(avctx, AV_LOG_ERROR, "AC EOB marker is absent pos=%d\n", mb->pos);

            block += 64;

            mb++;

        }

    }



    /* compute idct and place blocks */

    block = &sblock[0][0];

    mb    = mb_data;

    for (mb_index = 0; mb_index < 5; mb_index++) {

        dv_calculate_mb_xy(s, work_chunk, mb_index, &mb_x, &mb_y);



        /* idct_put'ting luminance */

        if ((s->sys->pix_fmt == PIX_FMT_YUV420P) ||

            (s->sys->pix_fmt == PIX_FMT_YUV411P && mb_x >= (704 / 8)) ||

            (s->sys->height >= 720 && mb_y != 134)) {

            y_stride = (s->picture.linesize[0] << ((!is_field_mode[mb_index]) * log2_blocksize));

        } else {

            y_stride = (2 << log2_blocksize);

        }

        y_ptr = s->picture.data[0] + ((mb_y * s->picture.linesize[0] + mb_x) << log2_blocksize);

        linesize = s->picture.linesize[0] << is_field_mode[mb_index];

        mb[0]    .idct_put(y_ptr                                   , linesize, block + 0*64);

        if (s->sys->video_stype == 4) { /* SD 422 */

            mb[2].idct_put(y_ptr + (1 << log2_blocksize)           , linesize, block + 2*64);

        } else {

            mb[1].idct_put(y_ptr + (1 << log2_blocksize)           , linesize, block + 1*64);

            mb[2].idct_put(y_ptr                         + y_stride, linesize, block + 2*64);

            mb[3].idct_put(y_ptr + (1 << log2_blocksize) + y_stride, linesize, block + 3*64);

        }

        mb += 4;

        block += 4*64;



        /* idct_put'ting chrominance */

        c_offset = (((mb_y >>  (s->sys->pix_fmt == PIX_FMT_YUV420P)) * s->picture.linesize[1] +

                     (mb_x >> ((s->sys->pix_fmt == PIX_FMT_YUV411P) ? 2 : 1))) << log2_blocksize);

        for (j = 2; j; j--) {

            uint8_t *c_ptr = s->picture.data[j] + c_offset;

            if (s->sys->pix_fmt == PIX_FMT_YUV411P && mb_x >= (704 / 8)) {

                  uint64_t aligned_pixels[64/8];

                  uint8_t *pixels = (uint8_t*)aligned_pixels;

                  uint8_t *c_ptr1, *ptr1;

                  int x, y;

                  mb->idct_put(pixels, 8, block);

                  for (y = 0; y < (1 << log2_blocksize); y++, c_ptr += s->picture.linesize[j], pixels += 8) {

                      ptr1   = pixels + (1 << (log2_blocksize - 1));

                      c_ptr1 = c_ptr + (s->picture.linesize[j] << log2_blocksize);

                      for (x = 0; x < (1 << (log2_blocksize - 1)); x++) {

                          c_ptr[x]  = pixels[x];

                          c_ptr1[x] = ptr1[x];

                      }

                  }

                  block += 64; mb++;

            } else {

                  y_stride = (mb_y == 134) ? (1 << log2_blocksize) :

                                             s->picture.linesize[j] << ((!is_field_mode[mb_index]) * log2_blocksize);

                  linesize = s->picture.linesize[j] << is_field_mode[mb_index];

                  (mb++)->    idct_put(c_ptr           , linesize, block); block += 64;

                  if (s->sys->bpm == 8) {

                      (mb++)->idct_put(c_ptr + y_stride, linesize, block); block += 64;

                  }

            }

        }

    }

    return 0;

}
