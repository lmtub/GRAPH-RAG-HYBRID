static inline int decode_mb(MDECContext *a, DCTELEM block[6][64]){

    int i;

    const int block_index[6]= {5,4,0,1,2,3};



    a->dsp.clear_blocks(block[0]);



    for(i=0; i<6; i++){

        if( mdec_decode_block_intra(a, block[ block_index[i] ], block_index[i]) < 0)

            return -1;

    }

    return 0;

}
