static int RENAME(swScale)(SwsContext *c, uint8_t* src[], int srcStride[], int srcSliceY,

             int srcSliceH, uint8_t* dst[], int dstStride[]){



	/* load a few things into local vars to make the code more readable? and faster */

	const int srcW= c->srcW;

	const int dstW= c->dstW;

	const int dstH= c->dstH;

	const int chrDstW= c->chrDstW;

	const int chrSrcW= c->chrSrcW;

	const int lumXInc= c->lumXInc;

	const int chrXInc= c->chrXInc;

	const int dstFormat= c->dstFormat;

	const int srcFormat= c->srcFormat;

	const int flags= c->flags;

	const int canMMX2BeUsed= c->canMMX2BeUsed;

	int16_t *vLumFilterPos= c->vLumFilterPos;

	int16_t *vChrFilterPos= c->vChrFilterPos;

	int16_t *hLumFilterPos= c->hLumFilterPos;

	int16_t *hChrFilterPos= c->hChrFilterPos;

	int16_t *vLumFilter= c->vLumFilter;

	int16_t *vChrFilter= c->vChrFilter;

	int16_t *hLumFilter= c->hLumFilter;

	int16_t *hChrFilter= c->hChrFilter;

	int32_t *lumMmxFilter= c->lumMmxFilter;

	int32_t *chrMmxFilter= c->chrMmxFilter;

	const int vLumFilterSize= c->vLumFilterSize;

	const int vChrFilterSize= c->vChrFilterSize;

	const int hLumFilterSize= c->hLumFilterSize;

	const int hChrFilterSize= c->hChrFilterSize;

	int16_t **lumPixBuf= c->lumPixBuf;

	int16_t **chrPixBuf= c->chrPixBuf;

	const int vLumBufSize= c->vLumBufSize;

	const int vChrBufSize= c->vChrBufSize;

	uint8_t *funnyYCode= c->funnyYCode;

	uint8_t *funnyUVCode= c->funnyUVCode;

	uint8_t *formatConvBuffer= c->formatConvBuffer;

	const int chrSrcSliceY= srcSliceY >> c->chrSrcVSubSample;

	const int chrSrcSliceH= -((-srcSliceH) >> c->chrSrcVSubSample);

	int lastDstY;

        uint8_t *pal=NULL;



	/* vars whch will change and which we need to storw back in the context */

	int dstY= c->dstY;

	int lumBufIndex= c->lumBufIndex;

	int chrBufIndex= c->chrBufIndex;

	int lastInLumBuf= c->lastInLumBuf;

	int lastInChrBuf= c->lastInChrBuf;

	

	if(isPacked(c->srcFormat)){

                pal= src[1];

		src[0]=

		src[1]=

		src[2]= src[0];

		srcStride[0]=

		srcStride[1]=

		srcStride[2]= srcStride[0];

	}

	srcStride[1]<<= c->vChrDrop;

	srcStride[2]<<= c->vChrDrop;



//	printf("swscale %X %X %X -> %X %X %X\n", (int)src[0], (int)src[1], (int)src[2],

//		(int)dst[0], (int)dst[1], (int)dst[2]);



#if 0 //self test FIXME move to a vfilter or something

{

static volatile int i=0;

i++;

if(srcFormat==PIX_FMT_YUV420P && i==1 && srcSliceH>= c->srcH)

	selfTest(src, srcStride, c->srcW, c->srcH);

i--;

}

#endif



//printf("sws Strides:%d %d %d -> %d %d %d\n", srcStride[0],srcStride[1],srcStride[2],

//dstStride[0],dstStride[1],dstStride[2]);



	if(dstStride[0]%8 !=0 || dstStride[1]%8 !=0 || dstStride[2]%8 !=0)

	{

		static int firstTime=1; //FIXME move this into the context perhaps

		if(flags & SWS_PRINT_INFO && firstTime)

		{

			av_log(c, AV_LOG_WARNING, "SwScaler: Warning: dstStride is not aligned!\n"

					"SwScaler:          ->cannot do aligned memory acesses anymore\n");

			firstTime=0;

		}

	}



	/* Note the user might start scaling the picture in the middle so this will not get executed

	   this is not really intended but works currently, so ppl might do it */

	if(srcSliceY ==0){

		lumBufIndex=0;

		chrBufIndex=0;

		dstY=0;	

		lastInLumBuf= -1;

		lastInChrBuf= -1;

	}



	lastDstY= dstY;



	for(;dstY < dstH; dstY++){

		unsigned char *dest =dst[0]+dstStride[0]*dstY;

		const int chrDstY= dstY>>c->chrDstVSubSample;

		unsigned char *uDest=dst[1]+dstStride[1]*chrDstY;

		unsigned char *vDest=dst[2]+dstStride[2]*chrDstY;



		const int firstLumSrcY= vLumFilterPos[dstY]; //First line needed as input

		const int firstChrSrcY= vChrFilterPos[chrDstY]; //First line needed as input

		const int lastLumSrcY= firstLumSrcY + vLumFilterSize -1; // Last line needed as input

		const int lastChrSrcY= firstChrSrcY + vChrFilterSize -1; // Last line needed as input



//printf("dstY:%d dstH:%d firstLumSrcY:%d lastInLumBuf:%d vLumBufSize: %d vChrBufSize: %d slice: %d %d vLumFilterSize: %d firstChrSrcY: %d vChrFilterSize: %d c->chrSrcVSubSample: %d\n",

// dstY, dstH, firstLumSrcY, lastInLumBuf, vLumBufSize, vChrBufSize, srcSliceY, srcSliceH, vLumFilterSize, firstChrSrcY, vChrFilterSize,  c->chrSrcVSubSample);

		//handle holes (FAST_BILINEAR & weird filters)

		if(firstLumSrcY > lastInLumBuf) lastInLumBuf= firstLumSrcY-1;

		if(firstChrSrcY > lastInChrBuf) lastInChrBuf= firstChrSrcY-1;

//printf("%d %d %d\n", firstChrSrcY, lastInChrBuf, vChrBufSize);

		ASSERT(firstLumSrcY >= lastInLumBuf - vLumBufSize + 1)

		ASSERT(firstChrSrcY >= lastInChrBuf - vChrBufSize + 1)



		// Do we have enough lines in this slice to output the dstY line

		if(lastLumSrcY < srcSliceY + srcSliceH && lastChrSrcY < -((-srcSliceY - srcSliceH)>>c->chrSrcVSubSample))

		{

			//Do horizontal scaling

			while(lastInLumBuf < lastLumSrcY)

			{

				uint8_t *s= src[0]+(lastInLumBuf + 1 - srcSliceY)*srcStride[0];

				lumBufIndex++;

//				printf("%d %d %d %d\n", lumBufIndex, vLumBufSize, lastInLumBuf,  lastLumSrcY);

				ASSERT(lumBufIndex < 2*vLumBufSize)

				ASSERT(lastInLumBuf + 1 - srcSliceY < srcSliceH)

				ASSERT(lastInLumBuf + 1 - srcSliceY >= 0)

//				printf("%d %d\n", lumBufIndex, vLumBufSize);

				RENAME(hyscale)(lumPixBuf[ lumBufIndex ], dstW, s, srcW, lumXInc,

						flags, canMMX2BeUsed, hLumFilter, hLumFilterPos, hLumFilterSize,

						funnyYCode, c->srcFormat, formatConvBuffer, 

						c->lumMmx2Filter, c->lumMmx2FilterPos, pal);

				lastInLumBuf++;

			}

			while(lastInChrBuf < lastChrSrcY)

			{

				uint8_t *src1= src[1]+(lastInChrBuf + 1 - chrSrcSliceY)*srcStride[1];

				uint8_t *src2= src[2]+(lastInChrBuf + 1 - chrSrcSliceY)*srcStride[2];

				chrBufIndex++;

				ASSERT(chrBufIndex < 2*vChrBufSize)

				ASSERT(lastInChrBuf + 1 - chrSrcSliceY < (chrSrcSliceH))

				ASSERT(lastInChrBuf + 1 - chrSrcSliceY >= 0)

				//FIXME replace parameters through context struct (some at least)



				if(!(isGray(srcFormat) || isGray(dstFormat)))

					RENAME(hcscale)(chrPixBuf[ chrBufIndex ], chrDstW, src1, src2, chrSrcW, chrXInc,

						flags, canMMX2BeUsed, hChrFilter, hChrFilterPos, hChrFilterSize,

						funnyUVCode, c->srcFormat, formatConvBuffer, 

						c->chrMmx2Filter, c->chrMmx2FilterPos, pal);

				lastInChrBuf++;

			}

			//wrap buf index around to stay inside the ring buffer

			if(lumBufIndex >= vLumBufSize ) lumBufIndex-= vLumBufSize;

			if(chrBufIndex >= vChrBufSize ) chrBufIndex-= vChrBufSize;

		}

		else // not enough lines left in this slice -> load the rest in the buffer

		{

/*		printf("%d %d Last:%d %d LastInBuf:%d %d Index:%d %d Y:%d FSize: %d %d BSize: %d %d\n",

			firstChrSrcY,firstLumSrcY,lastChrSrcY,lastLumSrcY,

			lastInChrBuf,lastInLumBuf,chrBufIndex,lumBufIndex,dstY,vChrFilterSize,vLumFilterSize,

			vChrBufSize, vLumBufSize);*/



			//Do horizontal scaling

			while(lastInLumBuf+1 < srcSliceY + srcSliceH)

			{

				uint8_t *s= src[0]+(lastInLumBuf + 1 - srcSliceY)*srcStride[0];

				lumBufIndex++;

				ASSERT(lumBufIndex < 2*vLumBufSize)

				ASSERT(lastInLumBuf + 1 - srcSliceY < srcSliceH)

				ASSERT(lastInLumBuf + 1 - srcSliceY >= 0)

				RENAME(hyscale)(lumPixBuf[ lumBufIndex ], dstW, s, srcW, lumXInc,

						flags, canMMX2BeUsed, hLumFilter, hLumFilterPos, hLumFilterSize,

						funnyYCode, c->srcFormat, formatConvBuffer, 

						c->lumMmx2Filter, c->lumMmx2FilterPos, pal);

				lastInLumBuf++;

			}

			while(lastInChrBuf+1 < (chrSrcSliceY + chrSrcSliceH))

			{

				uint8_t *src1= src[1]+(lastInChrBuf + 1 - chrSrcSliceY)*srcStride[1];

				uint8_t *src2= src[2]+(lastInChrBuf + 1 - chrSrcSliceY)*srcStride[2];

				chrBufIndex++;

				ASSERT(chrBufIndex < 2*vChrBufSize)

				ASSERT(lastInChrBuf + 1 - chrSrcSliceY < chrSrcSliceH)

				ASSERT(lastInChrBuf + 1 - chrSrcSliceY >= 0)



				if(!(isGray(srcFormat) || isGray(dstFormat)))

					RENAME(hcscale)(chrPixBuf[ chrBufIndex ], chrDstW, src1, src2, chrSrcW, chrXInc,

						flags, canMMX2BeUsed, hChrFilter, hChrFilterPos, hChrFilterSize,

						funnyUVCode, c->srcFormat, formatConvBuffer, 

						c->chrMmx2Filter, c->chrMmx2FilterPos, pal);

				lastInChrBuf++;

			}

			//wrap buf index around to stay inside the ring buffer

			if(lumBufIndex >= vLumBufSize ) lumBufIndex-= vLumBufSize;

			if(chrBufIndex >= vChrBufSize ) chrBufIndex-= vChrBufSize;

			break; //we can't output a dstY line so let's try with the next slice

		}



#ifdef HAVE_MMX

		b5Dither= dither8[dstY&1];

		g6Dither= dither4[dstY&1];

		g5Dither= dither8[dstY&1];

		r5Dither= dither8[(dstY+1)&1];

#endif

	    if(dstY < dstH-2)

	    {

		int16_t **lumSrcPtr= lumPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize;

		int16_t **chrSrcPtr= chrPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;

#ifdef HAVE_MMX

		int i;

            if(flags & SWS_ACCURATE_RND){

                        for(i=0; i<vLumFilterSize; i+=2){

                                lumMmxFilter[2*i+0]= (int32_t)lumSrcPtr[i  ];

                                lumMmxFilter[2*i+1]= (int32_t)lumSrcPtr[i+(vLumFilterSize>1)];

                                lumMmxFilter[2*i+2]=

                                lumMmxFilter[2*i+3]= vLumFilter[dstY*vLumFilterSize + i    ]

                                                + (vLumFilterSize>1 ? vLumFilter[dstY*vLumFilterSize + i + 1]<<16 : 0);

                        }

                        for(i=0; i<vChrFilterSize; i+=2){

                                chrMmxFilter[2*i+0]= (int32_t)chrSrcPtr[i  ];

                                chrMmxFilter[2*i+1]= (int32_t)chrSrcPtr[i+(vChrFilterSize>1)];

                                chrMmxFilter[2*i+2]=

                                chrMmxFilter[2*i+3]= vChrFilter[chrDstY*vChrFilterSize + i    ]

                                                + (vChrFilterSize>1 ? vChrFilter[chrDstY*vChrFilterSize + i + 1]<<16 : 0);

                        }

            }else{

		for(i=0; i<vLumFilterSize; i++)

		{

			lumMmxFilter[4*i+0]= (int32_t)lumSrcPtr[i];

			lumMmxFilter[4*i+1]= (uint64_t)lumSrcPtr[i] >> 32;

			lumMmxFilter[4*i+2]= 

			lumMmxFilter[4*i+3]= 

				((uint16_t)vLumFilter[dstY*vLumFilterSize + i])*0x10001;

		}

		for(i=0; i<vChrFilterSize; i++)

		{

			chrMmxFilter[4*i+0]= (int32_t)chrSrcPtr[i];


			chrMmxFilter[4*i+2]= 

			chrMmxFilter[4*i+3]= 

				((uint16_t)vChrFilter[chrDstY*vChrFilterSize + i])*0x10001;

		}

            }

#endif

		if(dstFormat == PIX_FMT_NV12 || dstFormat == PIX_FMT_NV21){

			const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

			if(dstY&chrSkipMask) uDest= NULL; //FIXME split functions in lumi / chromi

			RENAME(yuv2nv12X)(c,

				vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

				vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

				dest, uDest, dstW, chrDstW, dstFormat);

		}

		else if(isPlanarYUV(dstFormat) || isGray(dstFormat)) //YV12 like

		{

			const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

			if((dstY&chrSkipMask) || isGray(dstFormat)) uDest=vDest= NULL; //FIXME split functions in lumi / chromi

			if(vLumFilterSize == 1 && vChrFilterSize == 1) // Unscaled YV12

			{

				int16_t *lumBuf = lumPixBuf[0];

				int16_t *chrBuf= chrPixBuf[0];

				RENAME(yuv2yuv1)(lumBuf, chrBuf, dest, uDest, vDest, dstW, chrDstW);

			}

			else //General YV12

			{

				RENAME(yuv2yuvX)(c,

					vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

					vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

					dest, uDest, vDest, dstW, chrDstW);

			}

		}

		else

		{

			ASSERT(lumSrcPtr + vLumFilterSize - 1 < lumPixBuf + vLumBufSize*2);

			ASSERT(chrSrcPtr + vChrFilterSize - 1 < chrPixBuf + vChrBufSize*2);

			if(vLumFilterSize == 1 && vChrFilterSize == 2) //Unscaled RGB

			{

				int chrAlpha= vChrFilter[2*dstY+1];

				RENAME(yuv2packed1)(c, *lumSrcPtr, *chrSrcPtr, *(chrSrcPtr+1),

						 dest, dstW, chrAlpha, dstFormat, flags, dstY);

			}

			else if(vLumFilterSize == 2 && vChrFilterSize == 2) //BiLinear Upscale RGB

			{

				int lumAlpha= vLumFilter[2*dstY+1];

				int chrAlpha= vChrFilter[2*dstY+1];

                                lumMmxFilter[2]=

                                lumMmxFilter[3]= vLumFilter[2*dstY   ]*0x10001;

                                chrMmxFilter[2]=

                                chrMmxFilter[3]= vChrFilter[2*chrDstY]*0x10001;

				RENAME(yuv2packed2)(c, *lumSrcPtr, *(lumSrcPtr+1), *chrSrcPtr, *(chrSrcPtr+1),

						 dest, dstW, lumAlpha, chrAlpha, dstY);

			}

			else //General RGB

			{

				RENAME(yuv2packedX)(c,

					vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

					vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

					dest, dstW, dstY);

			}

		}

            }

	    else // hmm looks like we can't use MMX here without overwriting this array's tail

	    {

		int16_t **lumSrcPtr= lumPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize;

		int16_t **chrSrcPtr= chrPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;

		if(dstFormat == PIX_FMT_NV12 || dstFormat == PIX_FMT_NV21){

			const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

			if(dstY&chrSkipMask) uDest= NULL; //FIXME split functions in lumi / chromi

			yuv2nv12XinC(

				vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

				vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

				dest, uDest, dstW, chrDstW, dstFormat);

		}

		else if(isPlanarYUV(dstFormat) || isGray(dstFormat)) //YV12

		{

			const int chrSkipMask= (1<<c->chrDstVSubSample)-1;

			if((dstY&chrSkipMask) || isGray(dstFormat)) uDest=vDest= NULL; //FIXME split functions in lumi / chromi

			yuv2yuvXinC(

				vLumFilter+dstY*vLumFilterSize   , lumSrcPtr, vLumFilterSize,

				vChrFilter+chrDstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

				dest, uDest, vDest, dstW, chrDstW);

		}

		else

		{

			ASSERT(lumSrcPtr + vLumFilterSize - 1 < lumPixBuf + vLumBufSize*2);

			ASSERT(chrSrcPtr + vChrFilterSize - 1 < chrPixBuf + vChrBufSize*2);

			yuv2packedXinC(c, 

				vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

				vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

				dest, dstW, dstY);

		}

	    }

	}



#ifdef HAVE_MMX

	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");

#endif

	/* store changed local vars back in the context */

	c->dstY= dstY;

	c->lumBufIndex= lumBufIndex;

	c->chrBufIndex= chrBufIndex;

	c->lastInLumBuf= lastInLumBuf;

	c->lastInChrBuf= lastInChrBuf;



	return dstY - lastDstY;

}