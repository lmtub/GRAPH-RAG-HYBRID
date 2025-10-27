static void write_long(unsigned char *p,uint32_t v)

{

	p[0] = v>>24;

	p[1] = v>>16;

	p[2] = v>>8;

	p[3] = v;

}
