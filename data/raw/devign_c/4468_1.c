void yuv2rgb_altivec_init_tables (SwsContext *c, const int inv_table[4],int brightness,int contrast, int saturation)

{

  union {

  	signed short tmp[8] __attribute__ ((aligned(16)));

	vector signed short vec;

	} buf;



  buf.tmp[0] =  ( (0xffffLL) * contrast>>8 )>>9;			//cy

  buf.tmp[1] =  -256*brightness;					//oy

  buf.tmp[2] =  (inv_table[0]>>3) *(contrast>>16)*(saturation>>16);	//crv

  buf.tmp[3] =  (inv_table[1]>>3) *(contrast>>16)*(saturation>>16);	//cbu

  buf.tmp[4] = -((inv_table[2]>>1)*(contrast>>16)*(saturation>>16));	//cgu

  buf.tmp[5] = -((inv_table[3]>>1)*(contrast>>16)*(saturation>>16));	//cgv





  c->CSHIFT = (vector unsigned short)vec_splat_u16(2);

  c->CY  = vec_splat ((vector signed short)buf.vec, 0);

  c->OY  = vec_splat ((vector signed short)buf.vec, 1);

  c->CRV  = vec_splat ((vector signed short)buf.vec, 2);

  c->CBU  = vec_splat ((vector signed short)buf.vec, 3);

  c->CGU  = vec_splat ((vector signed short)buf.vec, 4);

  c->CGV  = vec_splat ((vector signed short)buf.vec, 5);

#if 0

{

int i;

char *v[6]={"cy","oy","crv","cbu","cgu","cgv"};

for (i=0; i<6;i++)

  printf("%s %d ", v[i],buf.tmp[i] );

  printf("\n");

}

#endif

 return;

}
