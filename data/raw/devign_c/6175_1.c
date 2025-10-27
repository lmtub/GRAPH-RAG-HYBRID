static void spatial_decompose97i(DWTELEM *buffer, int width, int height, int stride){

    int y;

    DWTELEM *b0= buffer + mirror(-4-1, height-1)*stride;

    DWTELEM *b1= buffer + mirror(-4  , height-1)*stride;

    DWTELEM *b2= buffer + mirror(-4+1, height-1)*stride;

    DWTELEM *b3= buffer + mirror(-4+2, height-1)*stride;



    for(y=-4; y<height; y+=2){

        DWTELEM *b4= buffer + mirror(y+3, height-1)*stride;

        DWTELEM *b5= buffer + mirror(y+4, height-1)*stride;



{START_TIMER

        if(b3 <= b5)     horizontal_decompose97i(b4, width);

        if(y+4 < height) horizontal_decompose97i(b5, width);

if(width>400){

STOP_TIMER("horizontal_decompose97i")

}}



{START_TIMER

        if(b3 <= b5) vertical_decompose97iH0(b3, b4, b5, width);

        if(b2 <= b4) vertical_decompose97iL0(b2, b3, b4, width);

        if(b1 <= b3) vertical_decompose97iH1(b1, b2, b3, width);

        if(b0 <= b2) vertical_decompose97iL1(b0, b1, b2, width);



if(width>400){

STOP_TIMER("vertical_decompose97i")

}}



        b0=b2;

        b1=b3;

        b2=b4;

        b3=b5;

    }

}
