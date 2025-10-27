static always_inline void dv_set_class_number(DCTELEM* blk, EncBlockInfo* bi, 

                                              const uint8_t* zigzag_scan, int bias)

{

    int i, area;

    int run;

    int classes[] = {12, 24, 36, 0xffff};



    run = 0;

    bi->mb[0] = blk[0]; 

    bi->cno = 0;

    for (area = 0; area < 4; area++) {

       bi->prev_run[area] = run;

       bi->bit_size[area] = 0;

       for (i=mb_area_start[area]; i<mb_area_start[area+1]; i++) {

          bi->mb[i] = (blk[zigzag_scan[i]] / 16);

          while ((bi->mb[i] ^ (bi->mb[i] >> 8)) > classes[bi->cno])

              bi->cno++;

       

          if (bi->mb[i]) {

              bi->bit_size[area] += dv_rl2vlc_size(run, bi->mb[i]);

	      run = 0;

          } else

              ++run;

       }

    }

    bi->bit_size[3] += 4; /* EOB marker */

    bi->cno += bias;

    

    if (bi->cno >= 3) { /* FIXME: we have to recreate bit_size[], prev_run[] */

        bi->cno = 3;

	for (i=1; i<64; i++)

	   bi->mb[i] /= 2;

    }

}
