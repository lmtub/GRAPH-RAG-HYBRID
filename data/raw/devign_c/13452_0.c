static inline int snake_search(MpegEncContext * s, int *best, int dmin,

                                       UINT8 *new_pic, UINT8 *old_pic, int pic_stride,

                                       int pred_x, int pred_y, UINT16 *mv_penalty, int quant,

                                       int xmin, int ymin, int xmax, int ymax, int shift)

{

    int dir=0;

    int c=1;

    static int x_dir[8]= {1,1,0,-1,-1,-1, 0, 1};

    static int y_dir[8]= {0,1,1, 1, 0,-1,-1,-1};

    int fails=0;

    int last_d[2]={dmin, dmin};



/*static int good=0;

static int bad=0;

static int point=0;



point++;

if(256*256*256*64%point==0)

{

    printf("%d %d %d\n", good, bad, point);

}*/



    for(;;){

        int x= best[0];

        int y= best[1];

        int d;

        x+=x_dir[dir];

        y+=y_dir[dir];

        if(x>=xmin && x<=xmax && y>=ymin && y<=ymax){

            d = pix_abs16x16(new_pic, old_pic + (x) + (y)*pic_stride, pic_stride);

            d += (mv_penalty[((x)<<shift)-pred_x] + mv_penalty[((y)<<shift)-pred_y])*quant;

        }else{

            d = dmin + 10000; //FIXME smarter boundary handling

        }

        if(d<dmin){

            best[0]=x;

            best[1]=y;

            dmin=d;



            if(last_d[1] - last_d[0] > last_d[0] - d) c= -c;

            dir+=c;



            fails=0;

//good++;

            last_d[1]=last_d[0];

            last_d[0]=d;

        }else{

//bad++;

            if(fails){

                if(fails>=3) return dmin;

            }else{

                c= -c;

            }

            dir+=c*2;

            fails++;

        }

        dir&=7;

    }

}
