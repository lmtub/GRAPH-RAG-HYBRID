static uint64_t get_vb(ByteIOContext *bc){

    uint64_t val=0;

    int i= get_v(bc);

    

    if(i>8)

        return UINT64_MAX;

    

    while(i--)

        val = (val<<8) + get_byte(bc);

    

//av_log(NULL, AV_LOG_DEBUG, "get_vb()= %lld\n", val);

    return val;

}
