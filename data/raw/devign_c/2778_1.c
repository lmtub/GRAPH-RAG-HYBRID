static void simpleCopy(SwsContext *c, uint8_t* srcParam[], int srcStrideParam[], int srcSliceY,

             int srcSliceH, uint8_t* dstParam[], int dstStride[]){



	int srcStride[3];

	uint8_t *src[3];

	uint8_t *dst[3];



	if(c->srcFormat == IMGFMT_I420){

		src[0]= srcParam[0];

		src[1]= srcParam[2];

		src[2]= srcParam[1];

		srcStride[0]= srcStrideParam[0];

		srcStride[1]= srcStrideParam[2];

		srcStride[2]= srcStrideParam[1];

	}

	else if(c->srcFormat==IMGFMT_YV12){

		src[0]= srcParam[0];

		src[1]= srcParam[1];

		src[2]= srcParam[2];

		srcStride[0]= srcStrideParam[0];

		srcStride[1]= srcStrideParam[1];

		srcStride[2]= srcStrideParam[2];

	}

	else if(isPacked(c->srcFormat) || isGray(c->srcFormat)){

		src[0]= srcParam[0];

		src[1]=

		src[2]= NULL;

		srcStride[0]= srcStrideParam[0];

		srcStride[1]=

		srcStride[2]= 0;

	}



	if(c->dstFormat == IMGFMT_I420){

		dst[0]= dstParam[0];

		dst[1]= dstParam[2];

		dst[2]= dstParam[1];

		

	}else{

		dst[0]= dstParam[0];

		dst[1]= dstParam[1];

		dst[2]= dstParam[2];

	}



	if(isPacked(c->srcFormat))

	{

		if(dstStride[0]==srcStride[0])

			memcpy(dst[0] + dstStride[0]*srcSliceY, src[0], srcSliceH*dstStride[0]);

		else

		{

			int i;

			uint8_t *srcPtr= src[0];

			uint8_t *dstPtr= dst[0] + dstStride[0]*srcSliceY;

			int length=0;



			/* universal length finder */

			while(length+c->srcW <= dstStride[0] 

			   && length+c->srcW <= srcStride[0]) length+= c->srcW;

			ASSERT(length!=0);



			for(i=0; i<srcSliceH; i++)

			{

				memcpy(dstPtr, srcPtr, length);

				srcPtr+= srcStride[0];

				dstPtr+= dstStride[0];

			}

		}

	}

	else 

	{ /* Planar YUV */

		int plane;

		for(plane=0; plane<3; plane++)

		{

			int length= plane==0 ? c->srcW  : ((c->srcW+1)>>1);

			int y=      plane==0 ? srcSliceY: ((srcSliceY+1)>>1);

			int height= plane==0 ? srcSliceH: ((srcSliceH+1)>>1);



			if(dstStride[plane]==srcStride[plane])

				memcpy(dst[plane] + dstStride[plane]*y, src[plane], height*dstStride[plane]);

			else

			{

				int i;

				uint8_t *srcPtr= src[plane];

				uint8_t *dstPtr= dst[plane] + dstStride[plane]*y;

				for(i=0; i<height; i++)

				{

					memcpy(dstPtr, srcPtr, length);

					srcPtr+= srcStride[plane];

					dstPtr+= dstStride[plane];

				}

			}

		}

	}

}
