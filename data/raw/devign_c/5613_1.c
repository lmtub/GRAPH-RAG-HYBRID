static void adx_decode_stereo(short *out,const unsigned char *in,PREV *prev)

{

	short tmp[32*2];

	int i;



	adx_decode(tmp   ,in   ,prev);

	adx_decode(tmp+32,in+18,prev+1);

	for(i=0;i<32;i++) {

		out[i*2]   = tmp[i];

		out[i*2+1] = tmp[i+32];

	}

}
