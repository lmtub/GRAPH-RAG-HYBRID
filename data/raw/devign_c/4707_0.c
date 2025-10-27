static int motion_inter_block (bit_buffer_t *bitbuf,

			       uint8_t *current, uint8_t *previous, int pitch,

			       svq1_pmv_t *motion, int x, int y) {

  uint8_t    *src;

  uint8_t    *dst;

  svq1_pmv_t  mv;

  svq1_pmv_t *pmv[3];

  int	      result;



  /* predict and decode motion vector */

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



  motion[0].x		= mv.x;

  motion[0].y		= mv.y;

  motion[(x / 8) + 2].x	= mv.x;

  motion[(x / 8) + 2].y	= mv.y;

  motion[(x / 8) + 3].x	= mv.x;

  motion[(x / 8) + 3].y	= mv.y;



  src = &previous[(x + (mv.x >> 1)) + (y + (mv.y >> 1))*pitch];

  dst = current;



  put_pixels_tab[((mv.y & 1) << 1) | (mv.x & 1)](dst,src,pitch,16);

  put_pixels_tab[((mv.y & 1) << 1) | (mv.x & 1)](dst+8,src+8,pitch,16);



  return 0;

}
