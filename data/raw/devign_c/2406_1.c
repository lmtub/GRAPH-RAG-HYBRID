static void init_uni_ac_vlc(RLTable *rl, uint8_t *uni_ac_vlc_len){

    int i;



    for(i=0; i<128; i++){

        int level= i-64;

        int run;



        for(run=0; run<64; run++){

            int len, bits, code;



            int alevel= FFABS(level);

            int sign= (level>>31)&1;



            if (alevel > rl->max_level[0][run])

                code= 111; /*rl->n*/

            else

                code= rl->index_run[0][run] + alevel - 1;



            if (code < 111 /* rl->n */) {

                /* store the vlc & sign at once */

                len=   rl->table_vlc[code][1]+1;

                bits= (rl->table_vlc[code][0]<<1) + sign;

            } else {

                len=  rl->table_vlc[111/*rl->n*/][1]+6;

                bits= rl->table_vlc[111/*rl->n*/][0]<<6;



                bits|= run;

                if (alevel < 128) {

                    bits<<=8; len+=8;

                    bits|= level & 0xff;

                } else {

                    bits<<=16; len+=16;

                    bits|= level & 0xff;

                    if (level < 0) {

                        bits|= 0x8001 + level + 255;

                    } else {

                        bits|= level & 0xffff;

                    }

                }

            }



            uni_ac_vlc_len [UNI_AC_ENC_INDEX(run, i)]= len;

        }

    }

}