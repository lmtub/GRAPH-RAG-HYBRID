static void reverse_dc_prediction(Vp3DecodeContext *s,

                                  int first_fragment,

                                  int fragment_width,

                                  int fragment_height)

{



#define PUL 8

#define PU 4

#define PUR 2

#define PL 1



    int x, y;

    int i = first_fragment;



    int predicted_dc;



    /* DC values for the left, up-left, up, and up-right fragments */

    int vl, vul, vu, vur;



    /* indexes for the left, up-left, up, and up-right fragments */

    int l, ul, u, ur;



    /*

     * The 6 fields mean:

     *   0: up-left multiplier

     *   1: up multiplier

     *   2: up-right multiplier

     *   3: left multiplier

     */

    static const int predictor_transform[16][4] = {

        {  0,  0,  0,  0},

        {  0,  0,  0,128},        // PL

        {  0,  0,128,  0},        // PUR

        {  0,  0, 53, 75},        // PUR|PL

        {  0,128,  0,  0},        // PU

        {  0, 64,  0, 64},        // PU|PL

        {  0,128,  0,  0},        // PU|PUR

        {  0,  0, 53, 75},        // PU|PUR|PL

        {128,  0,  0,  0},        // PUL

        {  0,  0,  0,128},        // PUL|PL

        { 64,  0, 64,  0},        // PUL|PUR

        {  0,  0, 53, 75},        // PUL|PUR|PL

        {  0,128,  0,  0},        // PUL|PU

       {-104,116,  0,116},        // PUL|PU|PL

        { 24, 80, 24,  0},        // PUL|PU|PUR

       {-104,116,  0,116}         // PUL|PU|PUR|PL

    };



    /* This table shows which types of blocks can use other blocks for

     * prediction. For example, INTRA is the only mode in this table to

     * have a frame number of 0. That means INTRA blocks can only predict

     * from other INTRA blocks. There are 2 golden frame coding types;

     * blocks encoding in these modes can only predict from other blocks

     * that were encoded with these 1 of these 2 modes. */

    static const unsigned char compatible_frame[8] = {

        1,    /* MODE_INTER_NO_MV */

        0,    /* MODE_INTRA */

        1,    /* MODE_INTER_PLUS_MV */

        1,    /* MODE_INTER_LAST_MV */

        1,    /* MODE_INTER_PRIOR_MV */

        2,    /* MODE_USING_GOLDEN */

        2,    /* MODE_GOLDEN_MV */

        1     /* MODE_INTER_FOUR_MV */

    };

    int current_frame_type;



    /* there is a last DC predictor for each of the 3 frame types */

    short last_dc[3];



    int transform = 0;



    vul = vu = vur = vl = 0;

    last_dc[0] = last_dc[1] = last_dc[2] = 0;



    /* for each fragment row... */

    for (y = 0; y < fragment_height; y++) {



        /* for each fragment in a row... */

        for (x = 0; x < fragment_width; x++, i++) {



            /* reverse prediction if this block was coded */

            if (s->all_fragments[i].coding_method != MODE_COPY) {



                current_frame_type =

                    compatible_frame[s->all_fragments[i].coding_method];



                transform= 0;

                if(x){

                    l= i-1;

                    vl = DC_COEFF(l);

                    if(FRAME_CODED(l) && COMPATIBLE_FRAME(l))

                        transform |= PL;

                }

                if(y){

                    u= i-fragment_width;

                    vu = DC_COEFF(u);

                    if(FRAME_CODED(u) && COMPATIBLE_FRAME(u))

                        transform |= PU;

                    if(x){

                        ul= i-fragment_width-1;

                        vul = DC_COEFF(ul);

                        if(FRAME_CODED(ul) && COMPATIBLE_FRAME(ul))

                            transform |= PUL;

                    }

                    if(x + 1 < fragment_width){

                        ur= i-fragment_width+1;

                        vur = DC_COEFF(ur);

                        if(FRAME_CODED(ur) && COMPATIBLE_FRAME(ur))

                            transform |= PUR;

                    }

                }



                if (transform == 0) {



                    /* if there were no fragments to predict from, use last

                     * DC saved */

                    predicted_dc = last_dc[current_frame_type];

                } else {



                    /* apply the appropriate predictor transform */

                    predicted_dc =

                        (predictor_transform[transform][0] * vul) +

                        (predictor_transform[transform][1] * vu) +

                        (predictor_transform[transform][2] * vur) +

                        (predictor_transform[transform][3] * vl);



                    predicted_dc /= 128;



                    /* check for outranging on the [ul u l] and

                     * [ul u ur l] predictors */

                    if ((transform == 13) || (transform == 15)) {

                        if (FFABS(predicted_dc - vu) > 128)

                            predicted_dc = vu;

                        else if (FFABS(predicted_dc - vl) > 128)

                            predicted_dc = vl;

                        else if (FFABS(predicted_dc - vul) > 128)

                            predicted_dc = vul;

                    }

                }



                /* at long last, apply the predictor */

                if(s->coeffs[i].index){

                    *s->next_coeff= s->coeffs[i];

                    s->coeffs[i].index=0;

                    s->coeffs[i].coeff=0;

                    s->coeffs[i].next= s->next_coeff++;

                }

                s->coeffs[i].coeff += predicted_dc;

                /* save the DC */

                last_dc[current_frame_type] = DC_COEFF(i);

                if(DC_COEFF(i) && !(s->coeff_counts[i]&127)){

                    s->coeff_counts[i]= 129;

//                    s->all_fragments[i].next_coeff= s->next_coeff;

                    s->coeffs[i].next= s->next_coeff;

                    (s->next_coeff++)->next=NULL;

                }

            }

        }

    }

}
