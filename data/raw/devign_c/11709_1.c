static void quantize(SnowContext *s, SubBand *b, DWTELEM *src, int stride, int bias){

    const int level= b->level;

    const int w= b->width;

    const int h= b->height;

    const int qlog= clip(s->qlog + b->qlog, 0, 128);

    const int qmul= qexp[qlog&7]<<(qlog>>3);

    int x,y;



    assert(QROOT==8);



    bias= bias ? 0 : (3*qmul)>>3;

    

    if(!bias){

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                int i= src[x + y*stride]; 

                //FIXME use threshold

                //FIXME optimize

                //FIXME bias

                if(i>=0){

                    i<<= QEXPSHIFT;

                    i/= qmul;

                    src[x + y*stride]=  i;

                }else{

                    i= -i;

                    i<<= QEXPSHIFT;

                    i/= qmul;

                    src[x + y*stride]= -i;

                }

            }

        }

    }else{

        for(y=0; y<h; y++){

            for(x=0; x<w; x++){

                int i= src[x + y*stride]; 

                

                //FIXME use threshold

                //FIXME optimize

                //FIXME bias

                if(i>=0){

                    i<<= QEXPSHIFT;

                    i= (i + bias) / qmul;

                    src[x + y*stride]=  i;

                }else{

                    i= -i;

                    i<<= QEXPSHIFT;

                    i= (i + bias) / qmul;

                    src[x + y*stride]= -i;

                }

            }

        }

    }

}
