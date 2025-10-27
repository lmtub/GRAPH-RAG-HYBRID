static void generate_codebook(RoqContext *enc, RoqTempdata *tempdata,

                              int *points, int inputCount, roq_cell *results,

                              int size, int cbsize)

{

    int i, j, k;

    int c_size = size*size/4;

    int *buf;

    int *codebook = av_malloc(6*c_size*cbsize*sizeof(int));

    int *closest_cb;



    if (size == 4)

        closest_cb = av_malloc(6*c_size*inputCount*sizeof(int));

    else

        closest_cb = tempdata->closest_cb2;



    ff_init_elbg(points, 6*c_size, inputCount, codebook, cbsize, 1, closest_cb, &enc->randctx);

    ff_do_elbg(points, 6*c_size, inputCount, codebook, cbsize, 1, closest_cb, &enc->randctx);



    if (size == 4)

        av_free(closest_cb);



    buf = codebook;

    for (i=0; i<cbsize; i++)

        for (k=0; k<c_size; k++) {

            for(j=0; j<4; j++)

                results->y[j] = *buf++;



            results->u =    (*buf++ + CHROMA_BIAS/2)/CHROMA_BIAS;

            results->v =    (*buf++ + CHROMA_BIAS/2)/CHROMA_BIAS;

            results++;

        }



    av_free(codebook);

}
