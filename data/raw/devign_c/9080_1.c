static void spatial_compose97i_dy(dwt_compose_t *cs, DWTELEM *buffer, int width, int height, int stride){

    int y = cs->y;

    DWTELEM *b0= cs->b0;

    DWTELEM *b1= cs->b1;

    DWTELEM *b2= cs->b2;

    DWTELEM *b3= cs->b3;

    DWTELEM *b4= buffer + mirror(y+3, height-1)*stride;

    DWTELEM *b5= buffer + mirror(y+4, height-1)*stride;



        if(stride == width && y+4 < height && 0){

            int x;

            for(x=0; x<width/2; x++)

                b5[x] += 64*2;

            for(; x<width; x++)

                b5[x] += 169*2;

        }



{START_TIMER

        if(b3 <= b5) vertical_compose97iL1(b3, b4, b5, width);

        if(b2 <= b4) vertical_compose97iH1(b2, b3, b4, width);

        if(b1 <= b3) vertical_compose97iL0(b1, b2, b3, width);

        if(b0 <= b2) vertical_compose97iH0(b0, b1, b2, width);

if(width>400){

STOP_TIMER("vertical_compose97i")}}



{START_TIMER

        if(y-1>=  0) horizontal_compose97i(b0, width);

        if(b0 <= b2) horizontal_compose97i(b1, width);

if(width>400 && b0 <= b2){

STOP_TIMER("horizontal_compose97i")}}



    cs->b0=b2;

    cs->b1=b3;

    cs->b2=b4;

    cs->b3=b5;

    cs->y += 2;

}
