static inline void dv_encode_video_segment(DVVideoContext *s, 

                                           uint8_t *dif, 

                                           const uint16_t *mb_pos_ptr)

{

    int mb_index, i, j, v;

    int mb_x, mb_y, c_offset, linesize; 

    uint8_t*  y_ptr;

    uint8_t*  data;

    uint8_t*  ptr;

    int       do_edge_wrap;

    DCTELEM   block[64] __align8;

    DCTELEM   sblock[5*6][64] __align8;

    EncBlockInfo  enc_blks[5*6];

    PutBitContext pbs[5*6];

    PutBitContext* pb; 

    EncBlockInfo* enc_blk;

    int       vs_bit_size = 0;

    int       qnos[5];

   

    enc_blk = &enc_blks[0];

    pb = &pbs[0];

    for(mb_index = 0; mb_index < 5; mb_index++) {

        v = *mb_pos_ptr++;

        mb_x = v & 0xff;

        mb_y = v >> 8;

        y_ptr = s->picture.data[0] + (mb_y * s->picture.linesize[0] * 8) + (mb_x * 8);

	c_offset = (s->sys->pix_fmt == PIX_FMT_YUV411P) ?

	           ((mb_y * s->picture.linesize[1] * 8) + ((mb_x >> 2) * 8)) :

		   (((mb_y >> 1) * s->picture.linesize[1] * 8) + ((mb_x >> 1) * 8));

	do_edge_wrap = 0;

	qnos[mb_index] = 15; /* No quantization */

        ptr = dif + mb_index*80 + 4;

        for(j = 0;j < 6; j++) {

            if (j < 4) {  /* Four Y blocks */

		/* NOTE: at end of line, the macroblock is handled as 420 */

		if (s->sys->pix_fmt == PIX_FMT_YUV411P && mb_x < (704 / 8)) {

                    data = y_ptr + (j * 8);

                } else {

                    data = y_ptr + ((j & 1) * 8) + ((j >> 1) * 8 * s->picture.linesize[0]);

                }

		linesize = s->picture.linesize[0];

            } else {      /* Cr and Cb blocks */

	        /* don't ask Fabrice why they inverted Cb and Cr ! */

	        data = s->picture.data[6 - j] + c_offset;

		linesize = s->picture.linesize[6 - j];

		if (s->sys->pix_fmt == PIX_FMT_YUV411P && mb_x >= (704 / 8))

		    do_edge_wrap = 1;

	    }	

            

	    /* Everything is set up -- now just copy data -> DCT block */

	    if (do_edge_wrap) {  /* Edge wrap copy: 4x16 -> 8x8 */

		uint8_t* d;

		DCTELEM *b = block;

	        for (i=0;i<8;i++) {

		   d = data + 8 * linesize;

		   b[0] = data[0]; b[1] = data[1]; b[2] = data[2]; b[3] = data[3];

                   b[4] =    d[0]; b[5] =    d[1]; b[6] =    d[2]; b[7] =    d[3];

		   data += linesize;

		   b += 8;

		}

	    } else {             /* Simple copy: 8x8 -> 8x8 */

	        s->get_pixels(block, data, linesize);

	    }

	  

            enc_blk->dct_mode = dv_guess_dct_mode(block);

	    enc_blk->mb = &sblock[mb_index*6+j][0];

	    enc_blk->area_q[0] = enc_blk->area_q[1] = enc_blk->area_q[2] = enc_blk->area_q[3] = 0;

	    enc_blk->partial_bit_count = 0;

	    enc_blk->partial_bit_buffer = 0;

	    enc_blk->cur_ac = 1;

	    

	    s->fdct[enc_blk->dct_mode](block);

	    

	    dv_set_class_number(block, enc_blk, 

	                        enc_blk->dct_mode ? ff_zigzag248_direct : ff_zigzag_direct,

				j/4*(j%2));

           

            init_put_bits(pb, ptr, block_sizes[j]/8);

	    put_bits(pb, 9, (uint16_t)(((enc_blk->mb[0] >> 3) - 1024) >> 2));

	    put_bits(pb, 1, enc_blk->dct_mode);

	    put_bits(pb, 2, enc_blk->cno);

	    

	    vs_bit_size += enc_blk->bit_size[0] + enc_blk->bit_size[1] +

	                   enc_blk->bit_size[2] + enc_blk->bit_size[3];

	    ++enc_blk;

	    ++pb;

	    ptr += block_sizes[j]/8;

        }

    }



    if (vs_total_ac_bits < vs_bit_size)

        dv_guess_qnos(&enc_blks[0], &qnos[0]);



    for (i=0; i<5; i++) {

       dif[i*80 + 3] = qnos[i];

    }



    /* First pass over individual cells only */

    for (j=0; j<5*6; j++)

       dv_encode_ac(&enc_blks[j], &pbs[j], 1);



    /* Second pass over each MB space */

    for (j=0; j<5*6; j++) {

       if (enc_blks[j].cur_ac < 65 || enc_blks[j].partial_bit_count)

           dv_encode_ac(&enc_blks[j], &pbs[(j/6)*6], 6);

    }



    /* Third and final pass over the whole vides segment space */

    for (j=0; j<5*6; j++) {

       if (enc_blks[j].cur_ac < 65 || enc_blks[j].partial_bit_count)

           dv_encode_ac(&enc_blks[j], &pbs[0], 6*5);

    }



    for (j=0; j<5*6; j++)

       flush_put_bits(&pbs[j]);

}
