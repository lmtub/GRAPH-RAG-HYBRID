void yuv2rgb_altivec_init_tables (SwsContext *c, const int inv_table[4])

{

  vector signed short CY, CRV, CBU, CGU, CGV, OY, Y0;

  int64_t crv __attribute__ ((aligned(16))) = inv_table[0];

  int64_t cbu __attribute__ ((aligned(16))) = inv_table[1];

  int64_t cgu __attribute__ ((aligned(16))) = inv_table[2];

  int64_t cgv __attribute__ ((aligned(16))) = inv_table[3];

  int64_t cy = (1<<16)-1;

  int64_t oy = 0;

  short tmp __attribute__ ((aligned(16)));



  if ((c->flags & SWS_CPU_CAPS_ALTIVEC) == 0)

    return;



  cy = (cy *c->contrast             )>>17;

  crv= (crv*c->contrast * c->saturation)>>32;

  cbu= (cbu*c->contrast * c->saturation)>>32;

  cgu= (cgu*c->contrast * c->saturation)>>32;

  cgv= (cgv*c->contrast * c->saturation)>>32;



  oy -= 256*c->brightness;



  tmp = cy;

  CY = vec_lde (0, &tmp);

  CY  = vec_splat (CY, 0);



  tmp = oy;

  OY = vec_lde (0, &tmp);

  OY  = vec_splat (OY, 0);



  tmp = crv>>3;

  CRV = vec_lde (0, &tmp);

  CRV  = vec_splat (CRV, 0);

  tmp = cbu>>3;

  CBU = vec_lde (0, &tmp);

  CBU  = vec_splat (CBU, 0);



  tmp = -(cgu>>1);

  CGU = vec_lde (0, &tmp);

  CGU  = vec_splat (CGU, 0);

  tmp = -(cgv>>1);

  CGV = vec_lde (0, &tmp);

  CGV  = vec_splat (CGV, 0);



  c->CSHIFT = (vector unsigned short)(2);

  c->CY = CY;

  c->OY = OY;

  c->CRV = CRV;

  c->CBU = CBU;

  c->CGU = CGU;

  c->CGV = CGV;



#if 0

  printf ("cy:  %hvx\n", CY);

  printf ("oy:  %hvx\n", OY);

  printf ("crv: %hvx\n", CRV);

  printf ("cbu: %hvx\n", CBU);

  printf ("cgv: %hvx\n", CGV);

  printf ("cgu: %hvx\n", CGU);

#endif



 return;

}
