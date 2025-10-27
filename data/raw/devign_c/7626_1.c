static int mlib_YUV2ABGR420_32(SwsContext *c, uint8_t* src[], int srcStride[], int srcSliceY,

             int srcSliceH, uint8_t* dst[], int dstStride[]){

    if(c->srcFormat == PIX_FMT_YUV422P){

	srcStride[1] *= 2;

	srcStride[2] *= 2;

    }



    assert(srcStride[1] == srcStride[2]);



    mlib_VideoColorYUV2ABGR420(dst[0]+srcSliceY*dstStride[0], src[0], src[1], src[2], c->dstW,

			     srcSliceH, dstStride[0], srcStride[0], srcStride[1]);

    return srcSliceH;

}
