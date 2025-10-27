static void generate_new_codebooks(RoqContext *enc, RoqTempdata *tempData)

{

    int i,j;

    RoqCodebooks *codebooks = &tempData->codebooks;

    int max = enc->width*enc->height/16;

    uint8_t mb2[3*4];

    roq_cell *results4 = av_malloc(sizeof(roq_cell)*MAX_CBS_4x4*4);

    uint8_t *yuvClusters=av_malloc(sizeof(int)*max*6*4);

    int *points = av_malloc(max*6*4*sizeof(int));

    int bias;



    /* Subsample YUV data */

    create_clusters(enc->frame_to_enc, enc->width, enc->height, yuvClusters);



    /* Cast to integer and apply chroma bias */

    for (i=0; i<max*24; i++) {

        bias = ((i%6)<4) ? 1 : CHROMA_BIAS;

        points[i] = bias*yuvClusters[i];

    }



    /* Create 4x4 codebooks */

    generate_codebook(enc, tempData, points, max, results4, 4, MAX_CBS_4x4);



    codebooks->numCB4 = MAX_CBS_4x4;



    tempData->closest_cb2 = av_malloc(max*4*sizeof(int));



    /* Create 2x2 codebooks */

    generate_codebook(enc, tempData, points, max*4, enc->cb2x2, 2, MAX_CBS_2x2);



    codebooks->numCB2 = MAX_CBS_2x2;



    /* Unpack 2x2 codebook clusters */

    for (i=0; i<codebooks->numCB2; i++)

        unpack_roq_cell(enc->cb2x2 + i, codebooks->unpacked_cb2 + i*2*2*3);



    /* Index all 4x4 entries to the 2x2 entries, unpack, and enlarge */

    for (i=0; i<codebooks->numCB4; i++) {

        for (j=0; j<4; j++) {

            unpack_roq_cell(&results4[4*i + j], mb2);

            index_mb(mb2, codebooks->unpacked_cb2, codebooks->numCB2,

                     &enc->cb4x4[i].idx[j], 2);

        }

        unpack_roq_qcell(codebooks->unpacked_cb2, enc->cb4x4 + i,

                         codebooks->unpacked_cb4 + i*4*4*3);

        enlarge_roq_mb4(codebooks->unpacked_cb4 + i*4*4*3,

                        codebooks->unpacked_cb4_enlarged + i*8*8*3);

    }



    av_free(yuvClusters);

    av_free(points);

    av_free(results4);

}
