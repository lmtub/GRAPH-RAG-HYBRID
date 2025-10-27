pp_context_t *pp_get_context(int width, int height, int cpuCaps){

	PPContext *c= memalign(32, sizeof(PPContext));

	int i;

	int stride= (width+15)&(~15); //assumed / will realloc if needed

        

	memset(c, 0, sizeof(PPContext));

	c->cpuCaps= cpuCaps;

	if(cpuCaps&PP_FORMAT){

		c->hChromaSubSample= cpuCaps&0x3;

		c->vChromaSubSample= (cpuCaps>>4)&0x3;

	}else{

		c->hChromaSubSample= 1;

		c->vChromaSubSample= 1;

	}



	reallocBuffers(c, width, height, stride);

        

	c->frameNum=-1;



	return c;

}
