static int decode_motion_vector (bit_buffer_t *bitbuf, svq1_pmv_t *mv, svq1_pmv_t **pmv) {

  uint32_t    bit_cache;

  vlc_code_t *vlc;

  int	      diff, sign;

  int	      i;



  for (i=0; i < 2; i++) {



    /* get motion code */

    bit_cache = get_bit_cache (bitbuf);



    if (!(bit_cache & 0xFFE00000))

      return -1;	/* invalid vlc code */



    if (bit_cache & 0x80000000) {

      diff = 0;



      /* flush bit */

      skip_bits(bitbuf,1);



    } else {

      if (bit_cache >= 0x06000000) {

        vlc = &motion_table_0[(bit_cache >> (32 - 7)) - 3];

      } else {

        vlc = &motion_table_1[(bit_cache >> (32 - 12)) - 2];

      }



      /* decode motion vector differential */

      sign = (int) (bit_cache << (vlc->length - 1)) >> 31;

      diff = (vlc->value ^ sign) - sign;



      /* flush bits */

      skip_bits(bitbuf,vlc->length);

    }



    /* add median of motion vector predictors and clip result */

    if (i == 1)

      mv->y = ((diff + MEDIAN(pmv[0]->y, pmv[1]->y, pmv[2]->y)) << 26) >> 26;

    else

      mv->x = ((diff + MEDIAN(pmv[0]->x, pmv[1]->x, pmv[2]->x)) << 26) >> 26;

  }



  return 0;

}
