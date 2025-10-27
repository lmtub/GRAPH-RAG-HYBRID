void ff_mpeg1_encode_init(MpegEncContext *s)

{

    static int done=0;



    common_init(s);



    if(!done){

        int f_code;

        int mv;

	int i;



        done=1;

        init_rl(&rl_mpeg1);



	for(i=0; i<64; i++)

	{

		mpeg1_max_level[0][i]= rl_mpeg1.max_level[0][i];

		mpeg1_index_run[0][i]= rl_mpeg1.index_run[0][i];

	}

        

        init_uni_ac_vlc(&rl_mpeg1, uni_mpeg1_ac_vlc_bits, uni_mpeg1_ac_vlc_len);



	/* build unified dc encoding tables */

	for(i=-255; i<256; i++)

	{

		int adiff, index;

		int bits, code;

		int diff=i;



		adiff = ABS(diff);

		if(diff<0) diff--;

		index = av_log2(2*adiff);



		bits= vlc_dc_lum_bits[index] + index;

		code= (vlc_dc_lum_code[index]<<index) + (diff & ((1 << index) - 1));

		mpeg1_lum_dc_uni[i+255]= bits + (code<<8);

		

		bits= vlc_dc_chroma_bits[index] + index;

		code= (vlc_dc_chroma_code[index]<<index) + (diff & ((1 << index) - 1));

		mpeg1_chr_dc_uni[i+255]= bits + (code<<8);

	}



        mv_penalty= av_mallocz( sizeof(uint8_t)*(MAX_FCODE+1)*(2*MAX_MV+1) );



        for(f_code=1; f_code<=MAX_FCODE; f_code++){

            for(mv=-MAX_MV; mv<=MAX_MV; mv++){

                int len;



                if(mv==0) len= mbMotionVectorTable[0][1];

                else{

                    int val, bit_size, range, code;



                    bit_size = f_code - 1;

                    range = 1 << bit_size;



                    val=mv;

                    if (val < 0) 

                        val = -val;

                    val--;

                    code = (val >> bit_size) + 1;

                    if(code<17){

                        len= mbMotionVectorTable[code][1] + 1 + bit_size;

                    }else{

                        len= mbMotionVectorTable[16][1] + 2 + bit_size;

                    }

                }



                mv_penalty[f_code][mv+MAX_MV]= len;

            }

        }

        



        for(f_code=MAX_FCODE; f_code>0; f_code--){

            for(mv=-(8<<f_code); mv<(8<<f_code); mv++){

                fcode_tab[mv+MAX_MV]= f_code;

            }

        }

    }

    s->me.mv_penalty= mv_penalty;

    s->fcode_tab= fcode_tab;

    if(s->codec_id == CODEC_ID_MPEG1VIDEO){

        s->min_qcoeff=-255;

        s->max_qcoeff= 255;

    }else{

        s->min_qcoeff=-2047;

        s->max_qcoeff= 2047;

    }

    s->intra_ac_vlc_length=

    s->inter_ac_vlc_length=

    s->intra_ac_vlc_last_length=

    s->inter_ac_vlc_last_length= uni_mpeg1_ac_vlc_len;

}
