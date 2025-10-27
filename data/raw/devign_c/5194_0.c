int MPV_common_init(MpegEncContext *s)

{

    int c_size, i;

    UINT8 *pict;



    s->dct_unquantize_h263 = dct_unquantize_h263_c;

    s->dct_unquantize_mpeg1 = dct_unquantize_mpeg1_c;

    s->dct_unquantize_mpeg2 = dct_unquantize_mpeg2_c;

        

#ifdef HAVE_MMX

    MPV_common_init_mmx(s);

#endif

    //setup default unquantizers (mpeg4 might change it later)

    if(s->out_format == FMT_H263)

        s->dct_unquantize = s->dct_unquantize_h263;

    else

        s->dct_unquantize = s->dct_unquantize_mpeg1;

    

    s->mb_width = (s->width + 15) / 16;

    s->mb_height = (s->height + 15) / 16;

    s->mb_num = s->mb_width * s->mb_height;

    s->linesize = s->mb_width * 16 + 2 * EDGE_WIDTH;



    for(i=0;i<3;i++) {

        int w, h, shift, pict_start;



        w = s->linesize;

        h = s->mb_height * 16 + 2 * EDGE_WIDTH;

        shift = (i == 0) ? 0 : 1;

        c_size = (w >> shift) * (h >> shift);

        pict_start = (w >> shift) * (EDGE_WIDTH >> shift) + (EDGE_WIDTH >> shift);



        pict = av_mallocz(c_size);

        if (pict == NULL)

            goto fail;

        s->last_picture_base[i] = pict;

        s->last_picture[i] = pict + pict_start;

    

        pict = av_mallocz(c_size);

        if (pict == NULL)

            goto fail;

        s->next_picture_base[i] = pict;

        s->next_picture[i] = pict + pict_start;

        

        if (s->has_b_frames || s->codec_id==CODEC_ID_MPEG4) {

        /* Note the MPEG4 stuff is here cuz of buggy encoders which dont set the low_delay flag but 

           do low-delay encoding, so we cant allways distinguish b-frame containing streams from low_delay streams */

            pict = av_mallocz(c_size);

            if (pict == NULL) 

                goto fail;

            s->aux_picture_base[i] = pict;

            s->aux_picture[i] = pict + pict_start;

        }

    }

    

    if (s->encoding) {

        int j;

        int mv_table_size= (s->mb_width+2)*(s->mb_height+2);



        /* Allocate MB type table */

        s->mb_type = av_mallocz(s->mb_num * sizeof(char));

        if (s->mb_type == NULL) {

            perror("malloc");

            goto fail;

        }

        

        s->mb_var = av_mallocz(s->mb_num * sizeof(INT16));

        if (s->mb_var == NULL) {

            perror("malloc");

            goto fail;

        }



        /* Allocate MV tables */

        s->p_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->p_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->last_p_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->last_p_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->b_forw_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->b_forw_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->b_back_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->b_back_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->b_bidir_forw_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->b_bidir_forw_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->b_bidir_back_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->b_bidir_back_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->b_direct_forw_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->b_direct_forw_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->b_direct_back_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->b_direct_back_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }

        s->b_direct_mv_table = av_mallocz(mv_table_size * 2 * sizeof(INT16));

        if (s->b_direct_mv_table == NULL) {

            perror("malloc");

            goto fail;

        }



        s->me_scratchpad = av_mallocz( s->linesize*16*3*sizeof(uint8_t));

        if (s->me_scratchpad == NULL) {

            perror("malloc");

            goto fail;

        }



        if(s->max_b_frames){

            for(j=0; j<REORDER_BUFFER_SIZE; j++){

                int i;

                for(i=0;i<3;i++) {

                    int w, h, shift;



                    w = s->linesize;

                    h = s->mb_height * 16;

                    shift = (i == 0) ? 0 : 1;

                    c_size = (w >> shift) * (h >> shift);



                    pict = av_mallocz(c_size);

                    if (pict == NULL)

                        goto fail;

                    s->picture_buffer[j][i] = pict;

                }

            }

        }

    }

    

    if (s->out_format == FMT_H263 || s->encoding) {

        int size;

        /* MV prediction */

        size = (2 * s->mb_width + 2) * (2 * s->mb_height + 2);

        s->motion_val = av_malloc(size * 2 * sizeof(INT16));

        if (s->motion_val == NULL)

            goto fail;

        memset(s->motion_val, 0, size * 2 * sizeof(INT16));

    }



    if (s->h263_pred || s->h263_plus) {

        int y_size, c_size, i, size;

        

        /* dc values */



        y_size = (2 * s->mb_width + 2) * (2 * s->mb_height + 2);

        c_size = (s->mb_width + 2) * (s->mb_height + 2);

        size = y_size + 2 * c_size;

        s->dc_val[0] = av_malloc(size * sizeof(INT16));

        if (s->dc_val[0] == NULL)

            goto fail;

        s->dc_val[1] = s->dc_val[0] + y_size;

        s->dc_val[2] = s->dc_val[1] + c_size;

        for(i=0;i<size;i++)

            s->dc_val[0][i] = 1024;



        /* ac values */

        s->ac_val[0] = av_mallocz(size * sizeof(INT16) * 16);

        if (s->ac_val[0] == NULL)

            goto fail;

        s->ac_val[1] = s->ac_val[0] + y_size;

        s->ac_val[2] = s->ac_val[1] + c_size;

        

        /* cbp values */

        s->coded_block = av_mallocz(y_size);

        if (!s->coded_block)

            goto fail;



        /* which mb is a intra block */

        s->mbintra_table = av_mallocz(s->mb_num);

        if (!s->mbintra_table)

            goto fail;

        memset(s->mbintra_table, 1, s->mb_num);

        

        /* divx501 bitstream reorder buffer */

        s->bitstream_buffer= av_mallocz(BITSTREAM_BUFFER_SIZE);

        if (!s->bitstream_buffer)

            goto fail;

    }

    /* default structure is frame */

    s->picture_structure = PICT_FRAME;



    /* init macroblock skip table */

    s->mbskip_table = av_mallocz(s->mb_num);

    if (!s->mbskip_table)

        goto fail;

    

    s->block= s->blocks[0];



    s->context_initialized = 1;

    return 0;

 fail:

    MPV_common_end(s);

    return -1;

}
