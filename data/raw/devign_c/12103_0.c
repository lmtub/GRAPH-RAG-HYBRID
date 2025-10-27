static void compute_default_clut(AVSubtitleRect *rect, int w, int h)

{

    uint8_t list[256] = {0};

    uint8_t list_inv[256];

    int counttab[256] = {0};

    int count, i, x, y;



#define V(x,y) rect->data[0][(x) + (y)*rect->linesize[0]]

    for (y = 0; y<h; y++) {

        for (x = 0; x<w; x++) {

            int v = V(x,y) + 1;

            int vl = x     ? V(x-1,y) + 1 : 0;

            int vr = x+1<w ? V(x+1,y) + 1 : 0;

            int vt = y     ? V(x,y-1) + 1 : 0;

            int vb = y+1<h ? V(x,y+1) + 1 : 0;

            counttab[v-1] += !!((v!=vl) + (v!=vr) + (v!=vt) + (v!=vb));

        }

    }

#define L(x,y) list[ rect->data[0][(x) + (y)*rect->linesize[0]] ]



    for (i = 0; i<256; i++) {

        int scoretab[256] = {0};

        int bestscore = 0;

        int bestv = 0;

        for (y = 0; y<h; y++) {

            for (x = 0; x<w; x++) {

                int v = rect->data[0][x + y*rect->linesize[0]];

                int l_m = list[v];

                int l_l = x     ? L(x-1, y) : 1;

                int l_r = x+1<w ? L(x+1, y) : 1;

                int l_t = y     ? L(x, y-1) : 1;

                int l_b = y+1<h ? L(x, y+1) : 1;

                int score;

                if (l_m)

                    continue;

                scoretab[v] += l_l + l_r + l_t + l_b;

                score = 1024LL*scoretab[v] / counttab[v];

                if (score > bestscore) {

                    bestscore = score;

                    bestv = v;

                }

            }

        }

        if (!bestscore)

            break;

        list    [ bestv ] = 1;

        list_inv[     i ] = bestv;

    }



    count = FFMAX(i - 1, 1);

    for (i--; i>=0; i--) {

        int v = i*255/count;

        AV_WN32(rect->data[1] + 4*list_inv[i], RGBA(v/2,v,v/2,v));

    }

}
