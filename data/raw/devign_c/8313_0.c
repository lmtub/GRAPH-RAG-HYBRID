static void create_cel_evals(RoqContext *enc, RoqTempdata *tempData)

{

    int n=0, x, y, i;



    tempData->cel_evals = av_malloc(enc->width*enc->height/64 * sizeof(CelEvaluation));



    /* Map to the ROQ quadtree order */

    for (y=0; y<enc->height; y+=16)

        for (x=0; x<enc->width; x+=16)

            for(i=0; i<4; i++) {

                tempData->cel_evals[n  ].sourceX = x + (i&1)*8;

                tempData->cel_evals[n++].sourceY = y + (i&2)*4;

            }

}
