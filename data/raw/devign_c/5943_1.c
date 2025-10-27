void ff_wmv2_idct_c(short * block){

    int i;



    for(i=0;i<64;i+=8){

        wmv2_idct_row(block+i);

    }

    for(i=0;i<8;i++){

        wmv2_idct_col(block+i);

    }

}
