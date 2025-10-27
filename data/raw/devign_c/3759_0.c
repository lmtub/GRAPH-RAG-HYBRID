static void allocate_buffers(FLACContext *s){

    int i;



    assert(s->max_blocksize);



    if(s->max_framesize == 0 && s->max_blocksize){

        s->max_framesize= (s->channels * s->bps * s->max_blocksize + 7)/ 8; //FIXME header overhead

    }



    for (i = 0; i < s->channels; i++)

    {

        s->decoded[i] = av_realloc(s->decoded[i], sizeof(int32_t)*s->max_blocksize);

    }



    s->bitstream= av_fast_realloc(s->bitstream, &s->allocated_bitstream_size, s->max_framesize);

}
