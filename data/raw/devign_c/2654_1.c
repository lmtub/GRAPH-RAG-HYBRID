static int compute_bit_allocation(AC3EncodeContext *s,

                                  uint8_t bap[NB_BLOCKS][AC3_MAX_CHANNELS][N/2],

                                  uint8_t encoded_exp[NB_BLOCKS][AC3_MAX_CHANNELS][N/2],

                                  uint8_t exp_strategy[NB_BLOCKS][AC3_MAX_CHANNELS],

                                  int frame_bits)

{

    int i, ch;

    int csnroffst, fsnroffst;

    uint8_t bap1[NB_BLOCKS][AC3_MAX_CHANNELS][N/2];

    static int frame_bits_inc[8] = { 0, 0, 2, 2, 2, 4, 2, 4 };



    /* init default parameters */

    s->sdecaycod = 2;

    s->fdecaycod = 1;

    s->sgaincod = 1;

    s->dbkneecod = 2;

    s->floorcod = 4;

    for(ch=0;ch<s->nb_all_channels;ch++) 

        s->fgaincod[ch] = 4;

    

    /* compute real values */

    s->bit_alloc.fscod = s->fscod;

    s->bit_alloc.halfratecod = s->halfratecod;

    s->bit_alloc.sdecay = sdecaytab[s->sdecaycod] >> s->halfratecod;

    s->bit_alloc.fdecay = fdecaytab[s->fdecaycod] >> s->halfratecod;

    s->bit_alloc.sgain = sgaintab[s->sgaincod];

    s->bit_alloc.dbknee = dbkneetab[s->dbkneecod];

    s->bit_alloc.floor = floortab[s->floorcod];

    

    /* header size */

    frame_bits += 65;

    // if (s->acmod == 2)

    //    frame_bits += 2;

    frame_bits += frame_bits_inc[s->acmod];



    /* audio blocks */

    for(i=0;i<NB_BLOCKS;i++) {

        frame_bits += s->nb_channels * 2 + 2; /* blksw * c, dithflag * c, dynrnge, cplstre */

        if (s->acmod == 2)

            frame_bits++; /* rematstr */

        frame_bits += 2 * s->nb_channels; /* chexpstr[2] * c */

	if (s->lfe)

	    frame_bits++; /* lfeexpstr */

        for(ch=0;ch<s->nb_channels;ch++) {

            if (exp_strategy[i][ch] != EXP_REUSE)

                frame_bits += 6 + 2; /* chbwcod[6], gainrng[2] */

        }

        frame_bits++; /* baie */

        frame_bits++; /* snr */

        frame_bits += 2; /* delta / skip */

    }

    frame_bits++; /* cplinu for block 0 */

    /* bit alloc info */

    /* sdcycod[2], fdcycod[2], sgaincod[2], dbpbcod[2], floorcod[3] */

    /* csnroffset[6] */

    /* (fsnoffset[4] + fgaincod[4]) * c */

    frame_bits += 2*4 + 3 + 6 + s->nb_all_channels * (4 + 3);



    /* auxdatae, crcrsv */

    frame_bits += 2;



    /* CRC */

    frame_bits += 16;



    /* now the big work begins : do the bit allocation. Modify the snr

       offset until we can pack everything in the requested frame size */



    csnroffst = s->csnroffst;

    while (csnroffst >= 0 && 

	   bit_alloc(s, bap, encoded_exp, exp_strategy, frame_bits, csnroffst, 0) < 0)

	csnroffst -= SNR_INC1;

    if (csnroffst < 0) {

	av_log(NULL, AV_LOG_ERROR, "Yack, Error !!!\n");

	return -1;

    }

    while ((csnroffst + SNR_INC1) <= 63 && 

           bit_alloc(s, bap1, encoded_exp, exp_strategy, frame_bits, 

                     csnroffst + SNR_INC1, 0) >= 0) {

        csnroffst += SNR_INC1;

        memcpy(bap, bap1, sizeof(bap1));

    }

    while ((csnroffst + 1) <= 63 && 

           bit_alloc(s, bap1, encoded_exp, exp_strategy, frame_bits, csnroffst + 1, 0) >= 0) {

        csnroffst++;

        memcpy(bap, bap1, sizeof(bap1));

    }



    fsnroffst = 0;

    while ((fsnroffst + SNR_INC1) <= 15 && 

           bit_alloc(s, bap1, encoded_exp, exp_strategy, frame_bits, 

                     csnroffst, fsnroffst + SNR_INC1) >= 0) {

        fsnroffst += SNR_INC1;

        memcpy(bap, bap1, sizeof(bap1));

    }

    while ((fsnroffst + 1) <= 15 && 

           bit_alloc(s, bap1, encoded_exp, exp_strategy, frame_bits, 

                     csnroffst, fsnroffst + 1) >= 0) {

        fsnroffst++;

        memcpy(bap, bap1, sizeof(bap1));

    }

    

    s->csnroffst = csnroffst;

    for(ch=0;ch<s->nb_all_channels;ch++)

        s->fsnroffst[ch] = fsnroffst;

#if defined(DEBUG_BITALLOC)

    {

        int j;



        for(i=0;i<6;i++) {

            for(ch=0;ch<s->nb_all_channels;ch++) {

                printf("Block #%d Ch%d:\n", i, ch);

                printf("bap=");

                for(j=0;j<s->nb_coefs[ch];j++) {

                    printf("%d ",bap[i][ch][j]);

                }

                printf("\n");

            }

        }

    }

#endif

    return 0;

}
