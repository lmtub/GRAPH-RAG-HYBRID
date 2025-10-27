static void *av_mallocz_static(unsigned int size)

{

    void *ptr = av_mallocz(size);



    if(ptr){

        array_static =av_fast_realloc(array_static, &allocated_static, sizeof(void*)*(last_static+1));

        if(!array_static)

            return NULL;

        array_static[last_static++] = ptr;

    }



    return ptr;

}
