static int motion_inter_4v_block (bit_buffer_t *bitbuf,

				  uint8_t *current, uint8_t *previous, int pitch,

				  svq1_pmv_t *motion,int x, int y) {

  uint8_t    *src;

  uint8_t    *dst;

  svq1_pmv_t  mv;

  svq1_pmv_t *pmv[4];

  int	      i, result;



  /* predict and decode motion vector (0) */

  pmv[0] = &motion[0];

  pmv[1] = &motion[(x / 8) + 2];

  pmv[2] = &motion[(x / 8) + 4];



  if (y == 0) {

    pmv[1] = pmv[0];

    pmv[2] = pmv[0];

  }



  result = decode_motion_vector (bitbuf, &mv, pmv);



  if (result != 0)

    return result;



  /* predict and decode motion vector (1) */

  pmv[0] = &mv;

  pmv[1] = &motion[(x / 8) + 3];



  if (y == 0) {

    pmv[1] = pmv[0];

    pmv[2] = pmv[0];

  }



  result = decode_motion_vector (bitbuf, &motion[0], pmv);



  if (result != 0)

    return result;



  /* predict and decode motion vector (2) */

  pmv[1] = &motion[0];

  pmv[2] = &motion[(x / 8) + 1];



  result = decode_motion_vector (bitbuf, &motion[(x / 8) + 2], pmv);



  if (result != 0)

    return result;



  /* predict and decode motion vector (3) */

  pmv[2] = &motion[(x / 8) + 2];

  pmv[3] = &motion[(x / 8) + 3];



  result = decode_motion_vector (bitbuf, pmv[3], pmv);



  if (result != 0)

    return result;



  /* form predictions */

  for (i=0; i < 4; i++) {

    src = &previous[(x + (pmv[i]->x >> 1)) + (y + (pmv[i]->y >> 1))*pitch];

    dst = current;



    put_pixels_tab[((pmv[i]->y & 1) << 1) | (pmv[i]->x & 1)](dst,src,pitch,8);



    /* select next block */

    if (i & 1) {

      current  += 8*(pitch - 1);

      previous += 8*(pitch - 1);

    } else {

      current  += 8;

      previous += 8;

    }

  }



  return 0;

}
