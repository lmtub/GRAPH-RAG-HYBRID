static void adx_encode(unsigned char *adx,const short *wav,PREV *prev)

{

	int scale;

	int i;

	int s0,s1,s2,d;

	int max=0;

	int min=0;

	int data[32];



	s1 = prev->s1;

	s2 = prev->s2;

	for(i=0;i<32;i++) {

		s0 = wav[i];

		d = ((s0<<14) - SCALE1*s1 + SCALE2*s2)/BASEVOL;

		data[i]=d;

		if (max<d) max=d;

		if (min>d) min=d;

		s2 = s1;

		s1 = s0;

	}

	prev->s1 = s1;

	prev->s2 = s2;



	/* -8..+7 */



	if (max==0 && min==0) {

		memset(adx,0,18);

		return;

	}



	if (max/7>-min/8) scale = max/7;

	else scale = -min/8;



	if (scale==0) scale=1;



	adx[0] = scale>>8;

	adx[1] = scale;



	for(i=0;i<16;i++) {

		adx[i+2] = ((data[i*2]/scale)<<4) | ((data[i*2+1]/scale)&0xf);

	}

}
